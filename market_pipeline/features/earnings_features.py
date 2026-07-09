"""
Kazanç açıklaması (earnings) yakınlık özellikleri.
- Gerçek tarihi zaman serisi — contamination yok
- yfinance earnings_dates: geçmiş ve planlanmış tarihler
- Options hacminin en büyük belirleyicisi (IV crush, pre-earnings premium)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR

EARNINGS_CACHE = DATA_DIR / "earnings_dates.json"


def _fetch_earnings_dates(symbol: str) -> list:
    """yfinance'den kazanç tarihlerini çek, timezone-naive list döndür."""
    try:
        ticker = yf.Ticker(symbol)
        ed = ticker.earnings_dates
        if ed is None or len(ed) == 0:
            return []
        dates = pd.to_datetime(ed.index)
        if hasattr(dates, 'tz') and dates.tz is not None:
            dates = dates.tz_convert('UTC').tz_localize(None)
        return sorted(str(d.date()) for d in dates)
    except Exception:
        return []


def load_earnings_cache(symbols: list, force: bool = False) -> dict:
    """Tüm semboller için earnings tarihlerini yükle veya indir."""
    cache = {}
    if not force and EARNINGS_CACHE.exists():
        try:
            cache = json.loads(EARNINGS_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    missing = [s for s in symbols if s not in cache]
    if missing:
        print(f"  Earnings tarihleri: {len(missing)} hisse indiriliyor...")
        for sym in missing:
            cache[sym] = _fetch_earnings_dates(sym)
        EARNINGS_CACHE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return cache


def compute_earnings_proximity(
    symbol: str,
    trade_dates: pd.DatetimeIndex,
    earnings_dates: list,
) -> pd.DataFrame:
    """
    Kazanç açıklaması yakınlık feature'ları.

    Özellikler:
    - Earnings_days_to_next : kaç gün sonra açıklama (max 90)
    - Earnings_days_since   : son açıklamadan kaç gün geçti (max 90)
    - Pre_earnings_14d      : 14 gün içinde açıklama var mı (0/1)
    - Pre_earnings_30d      : 30 gün içinde açıklama var mı (0/1)
    - Post_earnings_5d      : son açıklamadan bu yana <=5 gün (0/1)
    - Earnings_cycle_pct    : çeyreğin neresindeyiz 0→1 (0=hemen sonra, 1=hemen önce)
    """
    if not earnings_dates:
        return pd.DataFrame(
            0.0,
            index=trade_dates,
            columns=["Earnings_days_to_next", "Earnings_days_since",
                     "Pre_earnings_14d", "Pre_earnings_30d",
                     "Post_earnings_5d", "Earnings_cycle_pct"]
        )

    ts = sorted(pd.Timestamp(d) for d in earnings_dates)
    records = []
    for date in trade_dates:
        future = [e for e in ts if e > date]
        past   = [e for e in ts if e <= date]

        days_to_next   = int((future[0] - date).days) if future else 90
        days_since_last = int((date - past[-1]).days) if past else 90

        days_to_next    = min(days_to_next, 90)
        days_since_last = min(days_since_last, 90)

        # Çeyrek döngüsü: 0 = hemen açıklamadan sonra, 1 = açıklama arifesi
        cycle_total = days_since_last + days_to_next
        cycle_pct   = days_since_last / max(cycle_total, 1)

        records.append({
            "Earnings_days_to_next": days_to_next,
            "Earnings_days_since":   days_since_last,
            "Pre_earnings_14d":      float(0 < days_to_next <= 14),
            "Pre_earnings_30d":      float(0 < days_to_next <= 30),
            "Post_earnings_5d":      float(0 < days_since_last <= 5),
            "Earnings_cycle_pct":    float(cycle_pct),
        })

    return pd.DataFrame(records, index=trade_dates)


def add_earnings_to_features(
    feature_dict: dict,
    symbols: list,
    trade_dates: pd.DatetimeIndex,
    force: bool = False,
) -> dict:
    """
    Her hissenin feature DataFrame'ine kazanç yakınlık kolonları ekler.
    """
    earnings_cache = load_earnings_cache(symbols, force=force)

    for sym in symbols:
        if sym not in feature_dict:
            continue
        dates = earnings_cache.get(sym, [])
        prox  = compute_earnings_proximity(sym, trade_dates, dates)
        # Mevcut özellik DataFrame ile birleştir (index hizala)
        prox  = prox.reindex(feature_dict[sym].index).fillna(0.0)
        for col in prox.columns:
            feature_dict[sym][col] = prox[col].values

    return feature_dict
