"""
Paralel fiyat + hacim indirici.
yfinance'in toplu indirme özelliğini kullanır (500 hisseyi birden çeker).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from config import DATA_DIR, PERIOD, INTERVAL, MIN_HISTORY, MACRO_TICKERS, SECTOR_ETFS

PRICE_CACHE = DATA_DIR / "prices.parquet"
VOLUME_CACHE = DATA_DIR / "volumes.parquet"
MACRO_CACHE  = DATA_DIR / "macro.parquet"


def download_batch(symbols: list[str], batch_size: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sembolleri toplu indirir. (close, volume) döndürür.
    Hatalı semboller atlanır.
    """
    closes  = {}
    volumes = {}

    for i in tqdm(range(0, len(symbols), batch_size), desc="  Batch indirme"):
        batch = symbols[i: i + batch_size]
        try:
            raw = yf.download(
                batch,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            print(f"  Batch hatasi {i}-{i+batch_size}: {e}")
            continue

        if len(batch) == 1:
            sym = batch[0]
            try:
                c = raw["Close"].squeeze().dropna()
                v = raw["Volume"].squeeze().dropna()
                if len(c) >= MIN_HISTORY:
                    closes[sym]  = c
                    volumes[sym] = v
            except Exception:
                pass
        else:
            # yfinance yeni versiyonu: MultiIndex (Ticker, Price)
            for sym in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        # Yeni format: raw[(sym, "Close")] veya raw[sym]["Close"]
                        if sym in raw.columns.get_level_values(0):
                            c = raw[sym]["Close"].dropna()
                            v = raw[sym]["Volume"].dropna()
                        else:
                            continue
                    else:
                        # Eski format: raw["Close"][sym]
                        c = raw["Close"][sym].dropna()
                        v = raw["Volume"][sym].dropna()
                    if len(c) >= MIN_HISTORY:
                        closes[sym]  = c
                        volumes[sym] = v
                except Exception:
                    pass

    close_df  = pd.DataFrame(closes)
    volume_df = pd.DataFrame(volumes)
    close_df.index  = pd.to_datetime(close_df.index)
    volume_df.index = pd.to_datetime(volume_df.index)
    return close_df, volume_df


def download_prices(symbols: list[str], force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fiyat ve hacim verisi (önbellekli)."""
    if not force and PRICE_CACHE.exists():
        print(f"  Onbellekten yukleniyor: {PRICE_CACHE.name}")
        close_df  = pd.read_parquet(PRICE_CACHE)
        volume_df = pd.read_parquet(VOLUME_CACHE)
        return close_df, volume_df

    print(f"  {len(symbols)} sembol indiriliyor...")
    close_df, volume_df = download_batch(symbols)
    close_df.to_parquet(PRICE_CACHE)
    volume_df.to_parquet(VOLUME_CACHE)
    print(f"  Kaydedildi: {close_df.shape[1]} hisse, {len(close_df)} gun")
    return close_df, volume_df


def download_macro(force: bool = False) -> pd.DataFrame:
    """Makro ve sektör ETF verileri."""
    if not force and MACRO_CACHE.exists():
        return pd.read_parquet(MACRO_CACHE)

    all_tickers = list(MACRO_TICKERS.keys()) + list(SECTOR_ETFS.keys())
    print(f"  Makro veri: {len(all_tickers)} ticker")
    raw = yf.download(
        all_tickers, period=PERIOD, interval=INTERVAL,
        auto_adjust=True, progress=False, threads=True
    )
    frames = {}
    for sym in all_tickers:
        try:
            col_sym = sym.replace("^", "")
            if isinstance(raw.columns, pd.MultiIndex):
                # yfinance format: Level 0 = price_type, Level 1 = ticker
                if sym in raw.columns.get_level_values(1):
                    s = raw["Close"][sym].dropna()
                elif col_sym in raw.columns.get_level_values(1):
                    s = raw["Close"][col_sym].dropna()
                else:
                    continue
            else:
                s = raw["Close"][sym].dropna()
            s.name = col_sym
            frames[col_sym] = s
        except Exception:
            pass

    macro_df = pd.DataFrame(frames)
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df.to_parquet(MACRO_CACHE)
    return macro_df


if __name__ == "__main__":
    from tickers import fetch_sp500_tickers
    df = fetch_sp500_tickers()
    symbols = df["Symbol"].tolist()
    close, vol = download_prices(symbols)
    macro = download_macro()
    print(f"\nFiyat: {close.shape}  |  Makro: {macro.shape}")
