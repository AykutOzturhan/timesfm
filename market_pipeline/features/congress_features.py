"""
Kongre üyesi hisse işlem sinyalleri.
QuiverQuant ücretsiz endpoint veya Capitol Trades API kullanır.
Kongre alımları sektöre/hisseye yaklaşan düzenleyici veya kamu harcamasını sinyalleyebilir.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR

CONGRESS_CACHE = DATA_DIR / "congress_features.parquet"

# QuiverQuant ücretsiz endpoint (key gerektirmiyor bazı sembollerde)
QUIVER_URL = "https://api.quiverquant.com/beta/live/congresstrading/{symbol}"

# Capitol Trades alternatif endpoint
CAPITOL_URL = "https://www.capitoltrades.com/api/v1/trades?politician=all&asset={symbol}&page=1&pageSize=50"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research bot)",
    "Accept": "application/json",
}


def fetch_congress_snapshot(symbol: str) -> dict:
    """
    Kongre işlem verilerini çeker.
    Kaynak: QuiverQuant (ücretsiz) veya Capitol Trades.
    """
    result = {
        "congress_buy_count":   0.0,
        "congress_sell_count":  0.0,
        "congress_net_score":   0.0,   # (buy - sell) / (buy + sell + 1)
        "congress_has_activity": 0.0,
    }
    try:
        # QuiverQuant dene
        url = QUIVER_URL.format(symbol=symbol)
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                df = pd.DataFrame(data)
                if "Transaction" in df.columns:
                    buys  = df["Transaction"].str.upper().str.contains("BUY|PURCHASE", na=False).sum()
                    sells = df["Transaction"].str.upper().str.contains("SELL|SALE", na=False).sum()
                    total = buys + sells
                    result["congress_buy_count"]   = float(buys)
                    result["congress_sell_count"]  = float(sells)
                    result["congress_net_score"]   = (buys - sells) / (total + 1)
                    result["congress_has_activity"] = 1.0 if total > 0 else 0.0
                    return result
    except Exception:
        pass

    return result


def add_congress_to_features(
    feature_dict: dict,
    symbols: list,
    date_index: pd.DatetimeIndex,
    force: bool = False,
) -> dict:
    """
    feature_dict'e kongre işlem sinyalleri ekler.
    """
    if not force and CONGRESS_CACHE.exists():
        print("    Kongre verileri cache'ten yükleniyor...")
        cached = pd.read_parquet(CONGRESS_CACHE)
    else:
        print("    Kongre işlem verileri çekiliyor...")
        records = {}
        for i, sym in enumerate(symbols):
            snap = fetch_congress_snapshot(sym)
            records[sym] = snap
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(symbols)} tamamlandı")
        cached = pd.DataFrame(records).T
        cached.index.name = "symbol"
        cached.to_parquet(CONGRESS_CACHE)
        print(f"    Kongre verileri kaydedildi: {len(cached)} hisse")

    congress_cols = list(cached.columns)

    for sym in list(feature_dict.keys()):
        if sym not in cached.index:
            for col in congress_cols:
                feature_dict[sym][col] = 0.0
            continue
        row = cached.loc[sym]
        for col in congress_cols:
            feature_dict[sym][col] = float(row[col])

    return feature_dict
