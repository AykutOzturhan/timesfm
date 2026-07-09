"""
Temel (Fundamental) veriler: P/E, EPS büyümesi, gelir, borç/özkaynak.
yfinance info + quarterly financials.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from config import DATA_DIR

FUND_CACHE = DATA_DIR / "fundamentals.parquet"


def fetch_fundamentals(symbols: list[str], force: bool = False) -> pd.DataFrame:
    """
    Her sembol için anlık temel verileri çeker.
    Döndürür: sembol bazında DataFrame (tek satır/sembol)
    """
    if not force and FUND_CACHE.exists():
        print(f"  Temel veri onbellekten: {FUND_CACHE.name}")
        return pd.read_parquet(FUND_CACHE)

    rows = []
    for sym in tqdm(symbols, desc="  Temel veriler"):
        try:
            tk   = yf.Ticker(sym)
            info = tk.info or {}

            row = {
                "Symbol":           sym,
                "PE_trailing":      info.get("trailingPE"),
                "PE_forward":       info.get("forwardPE"),
                "PB":               info.get("priceToBook"),
                "PS":               info.get("priceToSalesTrailing12Months"),
                "EV_EBITDA":        info.get("enterpriseToEbitda"),
                "EPS_trailing":     info.get("trailingEps"),
                "EPS_forward":      info.get("forwardEps"),
                "Revenue_growth":   info.get("revenueGrowth"),      # YoY
                "Earnings_growth":  info.get("earningsGrowth"),     # YoY
                "Profit_margin":    info.get("profitMargins"),
                "ROE":              info.get("returnOnEquity"),
                "Debt_equity":      info.get("debtToEquity"),
                "Current_ratio":    info.get("currentRatio"),
                "Quick_ratio":      info.get("quickRatio"),
                "Beta":             info.get("beta"),
                "MarketCap":        info.get("marketCap"),
                "DividendYield":    info.get("dividendYield"),
                "52W_high":         info.get("fiftyTwoWeekHigh"),
                "52W_low":          info.get("fiftyTwoWeekLow"),
            }
            rows.append(row)
        except Exception:
            rows.append({"Symbol": sym})

    df = pd.DataFrame(rows).set_index("Symbol")

    # Sonsuz değerleri NaN yap, sonra sektör medyanıyla doldur
    df = df.replace([float("inf"), float("-inf")], np.nan)
    df.to_parquet(FUND_CACHE)
    print(f"  Kaydedildi: {len(df)} hisse, {df.shape[1]} metrik")
    return df


def add_percentile_rank(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her metriği 0-100 arasında percentile rank'a dönüştürür.
    Hisse bazlı karşılaştırma için kullanılır.
    """
    ranked = fund_df.copy()
    for col in ranked.columns:
        ranked[col] = ranked[col].rank(pct=True) * 100
    ranked.columns = [f"{c}_pct" for c in ranked.columns]
    return ranked


if __name__ == "__main__":
    from tickers import fetch_sp500_tickers
    tickers = fetch_sp500_tickers()
    syms = tickers["Symbol"].tolist()[:20]   # test için 20 hisse
    df = fetch_fundamentals(syms)
    print(df[["PE_trailing", "Beta", "Revenue_growth", "ROE"]].head(10))
