"""Sektör & göreli güç özellikleri."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORR_WINDOW, SECTOR_ETFS


def relative_strength(close_df: pd.DataFrame,
                      sector_etf_df: pd.DataFrame,
                      period: int = 20) -> pd.DataFrame:
    """
    Her hissenin kendi sektör ETF'ine göreli gücü.
    RS > 1 → sektörden güçlü, RS < 1 → zayıf.
    """
    ret_stock  = close_df.pct_change(period)
    rs_frames  = {}
    for etf in SECTOR_ETFS:
        if etf not in sector_etf_df.columns:
            continue
        ret_etf = sector_etf_df[etf].pct_change(period)
        # Her hisse için RS hesapla (broadcast)
        rs = ret_stock.subtract(ret_etf, axis=0)
        rs_frames[f"RS_{etf}"] = rs

    if not rs_frames:
        return pd.DataFrame()
    return pd.concat(rs_frames, axis=1)


def sector_momentum(sector_etf_df: pd.DataFrame,
                    periods: list[int] = [20, 60]) -> pd.DataFrame:
    """Sektör ETF'lerinin momentum skorları (hangi sektörler güçlü?)."""
    frames = {}
    for etf in SECTOR_ETFS:
        if etf not in sector_etf_df.columns:
            continue
        for p in periods:
            frames[f"{etf}_mom{p}"] = sector_etf_df[etf].pct_change(p)
    return pd.DataFrame(frames, index=sector_etf_df.index)


def rolling_beta(close_df: pd.DataFrame,
                 market: pd.Series,
                 window: int = 60) -> pd.DataFrame:
    """Her hissenin S&P 500'e karşı kayan beta'sı."""
    mkt_ret = market.pct_change()
    stk_ret = close_df.pct_change()

    def _beta(col: pd.Series) -> pd.Series:
        cov = col.rolling(window).cov(mkt_ret)
        var = mkt_ret.rolling(window).var()
        return cov / (var + 1e-9)

    beta_df = stk_ret.apply(_beta)
    beta_df.columns = [f"Beta_{c}" for c in beta_df.columns]
    return beta_df


def sector_rotation_signal(sector_etf_df: pd.DataFrame) -> pd.Series:
    """
    Sektör rotasyon skoru: son 20 günde hangi sektörler öne geçiyor?
    Risk-on (Teknoloji, Tüketim) vs risk-off (Utilities, Staples) oranı.
    """
    risk_on  = ["XLK", "XLY", "XLC"]
    risk_off = ["XLU", "XLP", "XLV"]

    on_ret  = sector_etf_df[[e for e in risk_on  if e in sector_etf_df.columns]].pct_change(20).mean(axis=1)
    off_ret = sector_etf_df[[e for e in risk_off if e in sector_etf_df.columns]].pct_change(20).mean(axis=1)
    return (on_ret - off_ret).rename("Rotation_signal")


def build_sector_features(close_df: pd.DataFrame,
                          macro_df: pd.DataFrame,
                          ticker_sector: dict[str, str]) -> pd.DataFrame:
    """
    Ana özellik oluşturma fonksiyonu.
    ticker_sector: {symbol: sector_name}
    Döndürür: (tarih x hisse, özellik) MultiIndex DataFrame
    """
    # Sektör ETF'lerini ayır
    etf_cols = [c for c in macro_df.columns if c in SECTOR_ETFS]
    sector_etf_df = macro_df[etf_cols] if etf_cols else pd.DataFrame(index=macro_df.index)

    # Her hisse için sektörüyle kıyasla
    features = {}

    # Sektör momentum (piyasa genelinde)
    sec_mom = sector_momentum(sector_etf_df)

    # Sektör rotasyon sinyali
    rot_sig = sector_rotation_signal(sector_etf_df)

    # SPY varsa beta hesapla
    spy = macro_df["SPY"] if "SPY" in macro_df.columns else None

    for sym in close_df.columns:
        sym_feats = {}

        # Sektör momentum (hisse sektörüne göre seç)
        sector = ticker_sector.get(sym, "")
        etf_for_sector = _sector_to_etf(sector)
        if etf_for_sector in sector_etf_df.columns:
            etf_price  = sector_etf_df[etf_for_sector]
            stk_ret20  = close_df[sym].pct_change(20)
            etf_ret20  = etf_price.pct_change(20)
            sym_feats["RS_sector_20d"] = stk_ret20 - etf_ret20

        # Rotasyon
        sym_feats["Rotation"] = rot_sig

        # Beta
        if spy is not None:
            mkt_ret = spy.pct_change()
            stk_ret = close_df[sym].pct_change()
            cov = stk_ret.rolling(60).cov(mkt_ret)
            var = mkt_ret.rolling(60).var()
            sym_feats["Beta_60d"] = cov / (var + 1e-9)

        features[sym] = pd.DataFrame(sym_feats, index=close_df.index)

    return features   # {sym: DataFrame}


def _sector_to_etf(sector: str) -> str:
    mapping = {
        "Information Technology":  "XLK",
        "Financials":              "XLF",
        "Energy":                  "XLE",
        "Health Care":             "XLV",
        "Consumer Discretionary":  "XLY",
        "Consumer Staples":        "XLP",
        "Utilities":               "XLU",
        "Industrials":             "XLI",
        "Materials":               "XLB",
        "Real Estate":             "XLRE",
        "Communication Services":  "XLC",
    }
    return mapping.get(sector, "SPY")
