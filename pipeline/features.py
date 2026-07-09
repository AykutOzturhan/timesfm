"""Aşama 2 — Teknik indikatörler ve özellik mühendisliği."""
import pandas as pd
import numpy as np


# ── Teknik İndikatörler ───────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename("RSI")


def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line
    return pd.DataFrame({
        "MACD":      macd_line,
        "MACD_sig":  sig_line,
        "MACD_hist": histogram,
    })


def bollinger(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid  = series.rolling(period).mean()
    band = series.rolling(period).std()
    upper = mid + std * band
    lower = mid - std * band
    pct_b = (series - lower) / (upper - lower + 1e-9)  # 0=alt, 1=üst
    return pd.DataFrame({
        "BB_upper":  upper,
        "BB_mid":    mid,
        "BB_lower":  lower,
        "BB_pct_b":  pct_b,
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().rename("ATR")


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    mu  = volume.rolling(period).mean()
    std = volume.rolling(period).std()
    return ((volume - mu) / (std + 1e-9)).rename("Vol_Z")


def returns(series: pd.Series, lags: list[int] = [1, 5, 20]) -> pd.DataFrame:
    out = {}
    for lag in lags:
        out[f"Ret_{lag}d"] = series.pct_change(lag)
    return pd.DataFrame(out)


def sector_correlation(target: pd.Series, market: pd.Series,
                       period: int = 60) -> pd.Series:
    """target ile market arasında kayan korelasyon."""
    return target.rolling(period).corr(market).rename("Corr_URA")


# ── Ana Özellik Tablosu ───────────────────────────────────────────────────

def build_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    prices_df: fetch_prices() çıktısı (UEC, URA, CCJ, DXY, VIX, TLT kolonları)
    Döndürür: birleşik özellik DataFrame (NaN'lar atılmış)
    """
    uec = prices_df["UEC"]
    features = pd.DataFrame(index=prices_df.index)

    # Fiyat seviyeleri (normalize)
    for col in ["UEC", "URA", "CCJ", "DXY", "VIX", "TLT"]:
        if col in prices_df.columns:
            features[col] = prices_df[col]

    # Teknik
    features = features.join(rsi(uec))
    features = features.join(macd(uec))
    features = features.join(bollinger(uec))
    features = features.join(returns(uec))

    # Göreceli güç
    features["Rel_URA"] = uec / prices_df["URA"]   # UEC/sektör oranı

    # Sektor korelasyon
    features = features.join(sector_correlation(uec, prices_df["URA"]))

    # VIX (risk iştahı — ters korelasyon beklenir)
    features["VIX_norm"] = (prices_df["VIX"] - prices_df["VIX"].rolling(60).mean()) \
                           / (prices_df["VIX"].rolling(60).std() + 1e-9)

    features = features.dropna()
    return features


if __name__ == "__main__":
    from data_collector import fetch_prices
    prices = fetch_prices()
    feats = build_features(prices)
    print(f"Ozellik tablosu: {feats.shape}")
    print(feats.columns.tolist())
    print(feats.tail(3).T)
