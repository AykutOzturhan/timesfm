"""
Options flow ve implied volatility özellikleri.
yfinance option chain'den IV, put/call oranı, IV rank hesaplar.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR

OPTIONS_CACHE = DATA_DIR / "options_features.parquet"


def _atm_iv(chain_df: pd.DataFrame, current_price: float) -> float:
    """En yakın strike'taki ortalama IV."""
    if chain_df.empty or current_price <= 0:
        return np.nan
    diff = (chain_df["strike"] - current_price).abs()
    idx  = diff.nsmallest(3).index
    iv   = chain_df.loc[idx, "impliedVolatility"].mean()
    return float(iv) if not np.isnan(iv) else np.nan


def fetch_options_snapshot(symbol: str) -> dict:
    """
    Tek hisse için anlık options metrikleri.
    OI bazlı put/call oranına ek olarak HACIM bazlı put/call oranı
    ve IV persentil rank da dahil edilir.
    """
    result = {
        "atm_iv_call":        np.nan,
        "atm_iv_put":         np.nan,
        "put_call_ratio":     np.nan,   # OI bazlı
        "put_call_vol_ratio": np.nan,   # Hacim bazlı — daha anlık sinyal
        "avg_call_iv":        np.nan,
        "avg_put_iv":         np.nan,
        "iv_skew":            np.nan,
        "iv_percentile":      np.nan,   # ATM IV'nin 30g penceredeki persentili
        "call_vol_spike":     np.nan,   # Anormal call hacmi (0/1)
        "put_vol_spike":      np.nan,   # Anormal put hacmi (0/1)
    }
    try:
        ticker = yf.Ticker(symbol)
        dates  = ticker.options
        if not dates:
            return result

        # Hem 30 günlük hem de mevcut expiryler
        today = pd.Timestamp.today()
        target_30d = today + pd.Timedelta(days=30)
        diffs = [(abs((pd.Timestamp(d) - target_30d).days), d) for d in dates]
        diffs.sort()
        exp_date = diffs[0][1]

        chain = ticker.option_chain(exp_date)
        calls = chain.calls.dropna(subset=["impliedVolatility", "strike"])
        puts  = chain.puts.dropna(subset=["impliedVolatility", "strike"])

        # Mevcut fiyat
        hist  = ticker.fast_info
        price = float(getattr(hist, "last_price", 0) or 0)
        if price <= 0:
            raw = yf.download(symbol, period="1d", progress=False)
            price = float(raw["Close"].iloc[-1]) if not raw.empty else 0

        if price > 0:
            result["atm_iv_call"] = _atm_iv(calls, price)
            result["atm_iv_put"]  = _atm_iv(puts, price)

        # Ortalama IV
        result["avg_call_iv"] = float(calls["impliedVolatility"].mean())
        result["avg_put_iv"]  = float(puts["impliedVolatility"].mean())

        # OI bazlı put/call
        call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 1
        put_oi  = puts["openInterest"].sum()  if "openInterest" in puts.columns  else 0
        if call_oi > 0:
            result["put_call_ratio"] = float(put_oi / call_oi)

        # Hacim bazlı put/call — OI'den çok daha anlık, günlük akış sinyali
        call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
        put_vol  = puts["volume"].sum()  if "volume" in puts.columns  else 0
        call_vol = call_vol if (call_vol and not np.isnan(call_vol)) else 0
        put_vol  = put_vol  if (put_vol  and not np.isnan(put_vol))  else 0
        if call_vol > 0:
            result["put_call_vol_ratio"] = float(put_vol / max(call_vol, 1))

        # IV skew: put ATM IV - call ATM IV (yüksek = aşağı yönlü korku)
        if not np.isnan(result["atm_iv_put"]) and not np.isnan(result["atm_iv_call"]):
            result["iv_skew"] = result["atm_iv_put"] - result["atm_iv_call"]

        # IV persentil: güncel ATM IV / (call_iv + put_iv) / 2 vs son 30 günlük
        # Proxy: avg_call_iv'nin kendi değeri / 52 hafta düzeltilmiş tahmin
        # Gerçek IV history yfinance'ta yok, bu yüzden avg vs atm farkını kullan
        atm_avg = ((result["atm_iv_call"] or 0) + (result["atm_iv_put"] or 0)) / 2
        avg_all  = ((result["avg_call_iv"] or 0) + (result["avg_put_iv"] or 0)) / 2
        if avg_all > 0:
            result["iv_percentile"] = float(np.clip(atm_avg / (avg_all + 1e-9), 0, 3))

        # Anormal hacim: toplam call/put hacmi >1000 işlem ise spike
        result["call_vol_spike"] = float(call_vol > 1000)
        result["put_vol_spike"]  = float(put_vol  > 1000)

    except Exception:
        pass

    return result


def build_options_features(
    symbols: list[str],
    force: bool = False,
) -> pd.DataFrame:
    """
    Tüm hisseler için options anlık görüntüsü.
    Günlük değişmez → bugünün değeriyle tüm geçmişe yay.
    Döndürür: DataFrame(index=symbol, columns=metrics)
    """
    if not force and OPTIONS_CACHE.exists():
        cached = pd.read_parquet(OPTIONS_CACHE)
        missing = [s for s in symbols if s not in cached.index]
        if not missing:
            return cached
    else:
        cached  = pd.DataFrame()
        missing = symbols

    print(f"  Options IV: {len(missing)} hisse çekiliyor...")
    rows = {}
    for sym in missing:
        rows[sym] = fetch_options_snapshot(sym)

    if rows:
        new_df = pd.DataFrame(rows).T
        new_df.index.name = "symbol"
        cached = pd.concat([cached, new_df]) if not cached.empty else new_df
        cached = cached[~cached.index.duplicated(keep="last")]
        cached.to_parquet(OPTIONS_CACHE)

    return cached


def add_options_to_features(
    feature_dict: dict,
    symbols: list[str],
    date_index: pd.DatetimeIndex,
    force: bool = False,
) -> dict:
    """
    Her hissenin feature DataFrame'ine options kolonları ekler.
    Eklenen kolonlar:
      - opt_atm_iv           : yakın vade ATM IV (ortalama call/put)
      - opt_put_call         : put/call open interest oranı
      - opt_put_call_vol     : put/call HACIM oranı (daha anlık akış sinyali)
      - opt_iv_skew          : put IV - call IV (negatif = bullish)
      - opt_avg_iv           : tüm strike ortalama IV
      - opt_iv_percentile    : ATM IV'nin ortalamaya oranı (yüksek = IV spike)
      - opt_call_spike       : anormal call hacmi (kısa squeeze öncüsü)
      - opt_put_spike        : anormal put hacmi (düşüş koruması talebi)
    """
    opts = build_options_features(symbols, force=force)

    for sym in symbols:
        if sym not in feature_dict or sym not in opts.index:
            continue
        row = opts.loc[sym]

        def _safe(v, default=0.0):
            if v is None:
                return default
            try:
                f = float(v)
                return default if (np.isnan(f) or np.isinf(f)) else f
            except (TypeError, ValueError):
                return default

        atm_call = _safe(row.get("atm_iv_call"))
        atm_put  = _safe(row.get("atm_iv_put"))
        atm_avg  = (atm_call + atm_put) / 2 if (atm_call > 0 or atm_put > 0) else 0.0

        feature_dict[sym]["opt_atm_iv"]   = atm_avg
        feature_dict[sym]["opt_put_call"] = _safe(row.get("put_call_ratio"))
        feature_dict[sym]["opt_iv_skew"]  = _safe(row.get("iv_skew"))
        avg_call = _safe(row.get("avg_call_iv"))
        avg_put  = _safe(row.get("avg_put_iv"))
        feature_dict[sym]["opt_avg_iv"]   = (avg_call + avg_put) / 2 if (avg_call > 0 or avg_put > 0) else 0.0
        # opt_put_call_vol, opt_iv_percentile, opt_call_spike, opt_put_spike:
        # static snapshot features — disabled (same issue: train has real value, backtest gets 0)

    return feature_dict


if __name__ == "__main__":
    syms = ["AAPL", "NVDA", "SPY", "UEC"]
    df   = build_options_features(syms, force=True)
    print(df.to_string())
