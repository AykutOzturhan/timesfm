"""
Haber duygu (sentiment) özellikleri.
yfinance haberlerini çeker, VADER ile skorlar, günlük ortalamaları hesaplar.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False

from config import DATA_DIR

SENTIMENT_CACHE = DATA_DIR / "news_sentiment.parquet"


def _score_text(text: str) -> float:
    """VADER compound skoru [-1, +1]."""
    if not _HAS_VADER or not text:
        return 0.0
    return _VADER.polarity_scores(text)["compound"]


def fetch_ticker_sentiment(symbol: str) -> pd.Series:
    """
    Verilen hisse için yfinance haberlerinden günlük sentiment skoru.
    Döndürür: pd.Series(index=date, values=compound_score)
    """
    try:
        ticker = yf.Ticker(symbol)
        news   = ticker.news
        if not news:
            return pd.Series(dtype=float, name=symbol)

        rows = []
        for item in news:
            # yfinance news formatı: providerPublishTime (unix timestamp) + title
            ts    = item.get("providerPublishTime", 0)
            title = item.get("title", "")
            summary = item.get("summary", "")
            text  = f"{title}. {summary}".strip()
            score = _score_text(text)
            date  = pd.Timestamp(ts, unit="s").normalize()
            rows.append({"date": date, "score": score})

        if not rows:
            return pd.Series(dtype=float, name=symbol)

        df = pd.DataFrame(rows)
        daily = df.groupby("date")["score"].mean()
        daily.name = symbol
        return daily
    except Exception:
        return pd.Series(dtype=float, name=symbol)


def build_sentiment_features(
    symbols: list[str],
    date_index: pd.DatetimeIndex,
    force: bool = False,
) -> pd.DataFrame:
    """
    Tüm hisseler için günlük sentiment matriksi.
    Sonuç: DataFrame(index=date, columns=symbols)
    Önbellekli — SENTIMENT_CACHE parquet dosyasına kaydeder.
    """
    if not force and SENTIMENT_CACHE.exists():
        cached = pd.read_parquet(SENTIMENT_CACHE)
        # Yeni hisselerin cache'de olmayan kısmını ekle
        missing = [s for s in symbols if s not in cached.columns]
        if not missing:
            return cached.reindex(date_index).ffill().fillna(0.0)
    else:
        cached = pd.DataFrame(index=date_index)
        missing = symbols

    print(f"  Haber sentiment: {len(missing)} hisse çekiliyor...")
    new_frames = {}
    for sym in missing:
        s = fetch_ticker_sentiment(sym)
        if not s.empty:
            new_frames[sym] = s

    if new_frames:
        new_df = pd.DataFrame(new_frames)
        new_df.index = pd.to_datetime(new_df.index)
        cached = pd.concat([cached, new_df], axis=1)
        # Sütun bazlı dedup
        cached = cached.loc[:, ~cached.columns.duplicated()]
        cached.to_parquet(SENTIMENT_CACHE)

    result = cached.reindex(date_index).ffill(limit=5).fillna(0.0)
    return result


def add_sentiment_to_features(
    feature_dict: dict,
    symbols: list[str],
    date_index: pd.DatetimeIndex,
) -> dict:
    """
    Her hissenin özellik DataFrame'ine 3 sentiment kolonu ekler:
      - sent_raw    : günlük ham VADER skoru
      - sent_ma5    : 5 günlük hareketli ortalama
      - sent_trend  : sent_ma5 - sent_ma20 (momentum)
    """
    sent_df = build_sentiment_features(symbols, date_index)

    for sym in symbols:
        if sym not in feature_dict or sym not in sent_df.columns:
            continue
        s = sent_df[sym].reindex(feature_dict[sym].index).ffill().fillna(0.0)
        feature_dict[sym]["sent_raw"]   = s.values
        feature_dict[sym]["sent_ma5"]   = s.rolling(5,  min_periods=1).mean().values
        feature_dict[sym]["sent_trend"] = (
            s.rolling(5, min_periods=1).mean() -
            s.rolling(20, min_periods=1).mean()
        ).values

    return feature_dict


if __name__ == "__main__":
    syms = ["AAPL", "NVDA", "UEC"]
    idx  = pd.bdate_range(end=datetime.today(), periods=30)
    df   = build_sentiment_features(syms, idx, force=True)
    print(df.tail())
    print(f"\nSentiment shape: {df.shape}")
    print(f"Sıfır olmayan değer oranı: {(df != 0).mean().mean():.1%}")
