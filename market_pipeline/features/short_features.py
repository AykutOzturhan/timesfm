"""
Short squeeze ve momentum sinyalleri.
yfinance fast_info.short_ratio ve major_holders'dan kısa pozisyon verisi.
Yüksek short ratio + pozitif momentum = squeeze potansiyeli.
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

SHORT_CACHE = DATA_DIR / "short_features.parquet"


def fetch_short_snapshot(symbol: str) -> dict:
    """
    Hisse için kısa pozisyon metrikleri.
    - short_ratio: days to cover (short interest / avg volume)
    - short_pct_float: float'ın yüzdesi olarak short pozisyon
    - squeeze_score: high short + positive momentum = squeeze riski
    """
    result = {
        "short_ratio":     0.0,
        "short_pct_float": 0.0,
        "squeeze_score":   0.0,
    }
    try:
        t = yf.Ticker(symbol)

        # Short ratio (days to cover)
        fi = t.fast_info
        sr = getattr(fi, "short_ratio", None)
        if sr is not None:
            try:
                result["short_ratio"] = float(sr) if np.isfinite(float(sr)) else 0.0
            except Exception:
                pass

        # Short % of float — major_holders formatına bak
        try:
            mh = t.major_holders
            if mh is not None and not mh.empty:
                # yfinance returns percentages in first column, labels in second
                # Look for "% of Shares Held by All Insider" and similar
                for _, row in mh.iterrows():
                    label = str(row.iloc[-1]).lower()
                    val_str = str(row.iloc[0]).replace("%", "").strip()
                    if "short" in label:
                        try:
                            result["short_pct_float"] = float(val_str) / 100
                        except Exception:
                            pass
        except Exception:
            pass

        # Squeeze score: short ratio * short_pct_float (normalized 0-1)
        # High short ratio AND high short pct = high squeeze potential
        squeeze = np.clip(result["short_ratio"] / 10.0, 0, 1) * \
                  np.clip(result["short_pct_float"] / 0.20, 0, 1)
        result["squeeze_score"] = float(squeeze)

    except Exception:
        pass

    return result


def add_short_to_features(
    feature_dict: dict,
    symbols: list,
    date_index: pd.DatetimeIndex,
    force: bool = False,
) -> dict:
    """
    feature_dict'e short squeeze sinyalleri ekler.
    """
    if not force and SHORT_CACHE.exists():
        print("    Short verileri cache'ten yükleniyor...")
        cached = pd.read_parquet(SHORT_CACHE)
    else:
        print("    Short interest verileri çekiliyor...")
        records = {}
        for i, sym in enumerate(symbols):
            records[sym] = fetch_short_snapshot(sym)
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(symbols)} tamamlandı")
        cached = pd.DataFrame(records).T
        cached.index.name = "symbol"
        cached.to_parquet(SHORT_CACHE)
        print(f"    Short verileri kaydedildi: {len(cached)} hisse")

    short_cols = list(cached.columns)

    for sym in list(feature_dict.keys()):
        if sym not in cached.index:
            for col in short_cols:
                feature_dict[sym][col] = 0.0
            continue
        row = cached.loc[sym]
        for col in short_cols:
            feature_dict[sym][col] = float(row[col])

    return feature_dict
