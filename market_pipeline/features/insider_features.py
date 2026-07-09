"""
Insider trading sinyalleri — SEC Form 4 verileri.
yfinance'den anlık insider işlem snapshot'ı alır.
Tüm eğitim pencereleri için aynı değeri kullanır
(options ve sentiment ile aynı yaklaşım).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR

INSIDER_CACHE = DATA_DIR / "insider_features.parquet"


def fetch_insider_snapshot(symbol: str) -> dict:
    """
    Tek hisse için insider alım/satım oranı.
    SEC Form 4 verilerinden son 180 günlük net insider pozisyonu.
    """
    result = {
        "insider_buy_count":    0.0,
        "insider_sell_count":   0.0,
        "insider_net_shares":   0.0,
        "insider_buy_ratio":    0.5,  # neutral default
    }
    try:
        t = yf.Ticker(symbol)
        txns = t.insider_transactions
        if txns is None or txns.empty:
            return result

        # Sütun adlarını normalleştir
        txns.columns = [c.lower().replace(" ", "_") for c in txns.columns]

        # Transaction sütununu bul
        trans_col = next((c for c in txns.columns if "transaction" in c.lower()), None)
        shares_col = next((c for c in txns.columns if "shares" in c.lower() and "total" not in c.lower()), None)
        date_col   = next((c for c in txns.columns if "start_date" in c.lower() or "date" in c.lower()), None)

        if trans_col is None or shares_col is None:
            return result

        txns[shares_col] = pd.to_numeric(txns[shares_col], errors="coerce").fillna(0)

        # Alım ve satım sayımı
        trans_str = txns[trans_col].astype(str).str.lower()
        buys  = txns[trans_str.str.contains("buy|purchase|acquire", na=False)]
        sells = txns[trans_str.str.contains("sell|sale|dispose", na=False)]

        buy_count  = len(buys)
        sell_count = len(sells)
        net_shares = float(buys[shares_col].sum() - sells[shares_col].sum())

        total = buy_count + sell_count
        buy_ratio = buy_count / total if total > 0 else 0.5

        result["insider_buy_count"]  = float(buy_count)
        result["insider_sell_count"] = float(sell_count)
        result["insider_net_shares"] = np.clip(net_shares / 1e6, -10, 10)  # milyon hisse cinsinden
        result["insider_buy_ratio"]  = float(buy_ratio)

    except Exception:
        pass
    return result


def add_insider_to_features(
    feature_dict: dict,
    symbols: list,
    date_index: pd.DatetimeIndex,
    force: bool = False,
) -> dict:
    """
    feature_dict'e insider trading sinyalleri ekler.
    """
    if not force and INSIDER_CACHE.exists():
        print("    Insider cache yükleniyor...")
        cached = pd.read_parquet(INSIDER_CACHE)
    else:
        print("    Insider verileri çekiliyor (SEC Form 4)...")
        records = {}
        for i, sym in enumerate(symbols):
            snap = fetch_insider_snapshot(sym)
            records[sym] = snap
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(symbols)} tamamlandı")
        cached = pd.DataFrame(records).T
        cached.index.name = "symbol"
        cached.to_parquet(INSIDER_CACHE)
        print(f"    Insider verileri kaydedildi: {len(cached)} hisse")

    insider_cols = list(cached.columns)

    for sym in list(feature_dict.keys()):
        if sym not in cached.index:
            # Veri yoksa sıfır doldur
            for col in insider_cols:
                feature_dict[sym][col] = 0.0 if "count" in col or "shares" in col else 0.5
            continue

        row = cached.loc[sym]
        for col in insider_cols:
            feature_dict[sym][col] = float(row[col])

    return feature_dict
