"""
Devlet sözleşmesi ve kamu harcaması sinyalleri.
USASpending.gov ücretsiz API'sinden sektör/hisse bazında kamu ihale verisi.
Savunma, sağlık, teknoloji (bulut), altyapı sektörlerine odaklanır.
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

GOVT_CACHE = DATA_DIR / "government_features.parquet"

USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

# NAICS → Sektör / ETF eşleştirme
NAICS_TO_SECTOR = {
    "336":  "Defense/Aerospace",     # Aircraft, missiles
    "334":  "Technology",             # Electronics, semiconductors
    "541":  "Technology",             # IT services, cloud
    "518":  "Technology",             # Cloud/data centers
    "621":  "Health Care",            # Health services
    "622":  "Health Care",            # Hospitals
    "325":  "Materials",              # Chemicals/pharma
    "237":  "Industrials",            # Infrastructure construction
    "333":  "Industrials",            # Machinery
    "211":  "Energy",                 # Oil/gas extraction
    "221":  "Utilities",              # Power generation
}

# Hisse → anahtar kelime eşleştirmesi (recipient name içerir mi?)
TICKER_KEYWORDS = {
    "MSFT":  ["microsoft"],
    "AMZN":  ["amazon", "aws"],
    "AAPL":  ["apple"],
    "GOOGL": ["google", "alphabet"],
    "META":  ["meta platform"],
    "RTX":   ["raytheon"],
    "LMT":   ["lockheed"],
    "BA":    ["boeing"],
    "GD":    ["general dynamics"],
    "NOC":   ["northrop"],
    "UNH":   ["unitedhealth", "united health"],
    "JNJ":   ["johnson & johnson", "j&j"],
    "CAT":   ["caterpillar"],
    "DE":    ["deere"],
    "XOM":   ["exxon"],
    "CVX":   ["chevron"],
}


def fetch_sector_spending() -> pd.DataFrame:
    """
    USASpending.gov'dan son 3 yıl sektör bazında ihale toplamı.
    Basit: birkaç büyük NAICS kodu için toplam tutarı çeker.
    """
    sector_totals = {}
    headers = {"Content-Type": "application/json"}

    for naics, sector in [("541", "Technology"), ("336", "Defense"),
                           ("621", "HealthCare"), ("237", "Infrastructure")]:
        try:
            payload = {
                "filters": {
                    "time_period": [{"start_date": "2022-01-01", "end_date": "2025-12-31"}],
                    "naics_codes": [naics],
                    "award_type_codes": ["A", "B", "C", "D"],
                },
                "fields": ["Award Amount"],
                "page": 1,
                "limit": 1,
                "sort": "Award Amount",
                "order": "desc",
            }
            resp = requests.post(USASPENDING_URL, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("page_metadata", {}).get("total", 0) or 0
                sector_totals[sector] = float(total)
        except Exception:
            sector_totals[sector] = 0.0

    return sector_totals


def fetch_company_contracts(symbol: str) -> dict:
    """
    Belirli bir şirkete verilen devlet sözleşmeleri.
    Sadece bilinen büyük devlet tedarikçileri için çalışır.
    """
    keywords = TICKER_KEYWORDS.get(symbol.upper(), [])
    result = {
        "govt_contract_count":  0.0,
        "govt_contract_value":  0.0,
        "is_govt_contractor":   0.0,
    }
    if not keywords:
        return result

    headers = {"Content-Type": "application/json"}
    total_value = 0.0
    total_count = 0

    for keyword in keywords[:1]:  # sadece ilk keyword, hız için
        try:
            payload = {
                "filters": {
                    "time_period": [{"start_date": "2023-01-01", "end_date": "2025-12-31"}],
                    "recipient_search_text": [keyword],
                    "award_type_codes": ["A", "B", "C", "D"],
                },
                "fields": ["Award Amount", "Recipient Name"],
                "page": 1,
                "limit": 5,
            }
            resp = requests.post(USASPENDING_URL, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                for r in results:
                    val = r.get("Award Amount") or 0
                    total_value += float(val)
                    total_count += 1
        except Exception:
            pass

    result["govt_contract_count"] = float(total_count)
    result["govt_contract_value"] = np.log1p(total_value / 1e6)  # log(million USD)
    result["is_govt_contractor"]  = 1.0 if total_count > 0 else 0.0
    return result


def add_government_to_features(
    feature_dict: dict,
    symbols: list,
    date_index: pd.DatetimeIndex,
    force: bool = False,
) -> dict:
    """
    feature_dict'e devlet sözleşme sinyalleri ekler.
    """
    if not force and GOVT_CACHE.exists():
        print("    Devlet sözleşme verileri cache'ten yükleniyor...")
        cached = pd.read_parquet(GOVT_CACHE)
    else:
        print("    Devlet sözleşme verileri çekiliyor (USASpending.gov)...")
        records = {}
        for i, sym in enumerate(symbols):
            snap = fetch_company_contracts(sym)
            records[sym] = snap
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(symbols)} tamamlandı")
        cached = pd.DataFrame(records).T
        cached.index.name = "symbol"
        cached.to_parquet(GOVT_CACHE)
        print(f"    Devlet sözleşme verileri kaydedildi: {len(cached)} hisse")

    govt_cols = list(cached.columns)

    for sym in list(feature_dict.keys()):
        if sym not in cached.index:
            for col in govt_cols:
                feature_dict[sym][col] = 0.0
            continue
        row = cached.loc[sym]
        for col in govt_cols:
            feature_dict[sym][col] = float(row[col])

    return feature_dict
