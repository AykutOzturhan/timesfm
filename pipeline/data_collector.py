"""Aşama 1 — Veri toplama: fiyat, hacim, haberler, macro."""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


TICKERS = {
    "UEC":  "Uranium Energy Corp (hedef)",
    "URA":  "Global X Uranium ETF (sektör)",
    "CCJ":  "Cameco Corp (rakip)",
    "DXY":  "US Dollar Index proxy (UUP ETF)",
    "VIX":  "Volatility Index (^VIX)",
    "TLT":  "20Y Treasury Bond ETF (risk iştahı)",
}

SYMBOL_MAP = {
    "UEC": "UEC",
    "URA": "URA",
    "CCJ": "CCJ",
    "DXY": "UUP",      # DXY'yi doğrudan çekmek zor, UUP proxy
    "VIX": "^VIX",
    "TLT": "TLT",
}


def fetch_prices(period: str = "3y") -> pd.DataFrame:
    """Tüm tickerlarin günlük kapanış fiyatlarını çeker."""
    frames = {}
    for name, sym in SYMBOL_MAP.items():
        try:
            raw = yf.download(sym, period=period, interval="1d",
                              progress=False, auto_adjust=True)
            close = raw["Close"].squeeze().dropna()
            close.name = name
            frames[name] = close
            print(f"  [{name}] {sym}: {len(close)} gün")
        except Exception as e:
            print(f"  [{name}] HATA: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df


def fetch_news(ticker: str = "UEC", max_items: int = 200) -> pd.DataFrame:
    """Yahoo Finance haber başlıklarını çeker."""
    tk = yf.Ticker(ticker)
    try:
        news = tk.news or []
    except Exception:
        news = []

    rows = []
    for item in news[:max_items]:
        content = item.get("content", {})
        title = content.get("title", "")
        pub_date = content.get("pubDate", "")
        summary = content.get("summary", "")
        try:
            dt = pd.to_datetime(pub_date, utc=True).tz_localize(None)
        except Exception:
            dt = None
        if title and dt is not None:
            rows.append({"date": dt.date(), "title": title, "summary": summary})

    if not rows:
        return pd.DataFrame(columns=["date", "title", "summary"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


if __name__ == "__main__":
    print("Fiyat verileri cekiliyor...")
    prices = fetch_prices()
    print(f"\nVeri sekli: {prices.shape}")
    print(prices.tail(3))

    print("\nHaberler cekiliyor...")
    news = fetch_news()
    print(f"Haber sayisi: {len(news)}")
    if len(news) > 0:
        print(news[["date", "title"]].tail(5).to_string())
