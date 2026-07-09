"""Aşama 3 — Haber sentiment analizi (VADER, FinBERT opsiyonel)."""
import pandas as pd
import numpy as np

# VADER: hafif, finansal metinlerde makul sonuç verir
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except ImportError:
    VADER_OK = False

# FinBERT: finansal metinler için eğitilmiş BERT (opsiyonel, ağır)
FINBERT_OK = False
try:
    from transformers import pipeline as hf_pipeline
    FINBERT_OK = True
except ImportError:
    pass


def score_vader(texts: list[str]) -> list[float]:
    """Her metin için -1 (negatif) ile +1 (pozitif) arası skor."""
    if not VADER_OK:
        return [0.0] * len(texts)
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]


def score_finbert(texts: list[str], batch_size: int = 16) -> list[float]:
    """FinBERT ile finansal sentiment skoru."""
    if not FINBERT_OK:
        print("  transformers yuklu degil, VADER'a dusuluyor.")
        return score_vader(texts)

    pipe = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=-1,           # CPU
        truncation=True,
        max_length=512,
    )
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        results = pipe(batch)
        for r in results:
            scores.append(label_map.get(r["label"].lower(), 0.0) * r["score"])
    return scores


def build_daily_sentiment(news_df: pd.DataFrame,
                          use_finbert: bool = False) -> pd.Series:
    """
    news_df: date, title, summary kolonları
    Döndürür: günlük ortalama sentiment skoru (pd.Series, index=date)
    """
    if news_df.empty:
        return pd.Series(dtype=float, name="Sentiment")

    texts = (news_df["title"] + ". " + news_df["summary"].fillna("")).tolist()
    fn = score_finbert if use_finbert else score_vader

    if not VADER_OK and not use_finbert:
        print("  VADER kurulu degil. pip install vaderSentiment")
        news_df = news_df.copy()
        news_df["score"] = 0.0
    else:
        news_df = news_df.copy()
        news_df["score"] = fn(texts)

    daily = news_df.groupby("date")["score"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily.name = "Sentiment"
    return daily


def merge_sentiment(features_df: pd.DataFrame,
                    sentiment: pd.Series,
                    fill_days: int = 3) -> pd.DataFrame:
    """
    Günlük sentiment'i features tablosuna ekler.
    Haber olmayan günler için son N günün ortalamasını ileri taşır.
    """
    df = features_df.copy()
    df["Sentiment"] = sentiment.reindex(df.index)
    # Son gelen haberin etkisi fill_days gün devam eder
    df["Sentiment"] = df["Sentiment"].ffill(limit=fill_days).fillna(0.0)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_collector import fetch_news
    from features import build_features, fetch_prices

    print("Haberler cekiliyor...")
    news = fetch_news("UEC")
    print(f"Toplam haber: {len(news)}")

    if len(news) > 0:
        sent = build_daily_sentiment(news, use_finbert=False)
        print(f"\nGunluk sentiment ornek:")
        print(sent.tail(10))
        print(f"\nOrtalama: {sent.mean():.3f}  |  Std: {sent.std():.3f}")
    else:
        print("Haber bulunamadi.")
