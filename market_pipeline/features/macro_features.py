"""Makro özellikler: yield spread, VIX rejimi, dolar trendi."""
import pandas as pd
import numpy as np


def yield_spread(macro_df: pd.DataFrame) -> pd.Series:
    """10Y - 3M Treasury spread (ters eğri = resesyon sinyali)."""
    if "TNX" in macro_df and "IRX" in macro_df:
        return (macro_df["TNX"] - macro_df["IRX"]).rename("Yield_spread")
    return pd.Series(dtype=float, name="Yield_spread")


def vix_regime(macro_df: pd.DataFrame) -> pd.DataFrame:
    """VIX seviyesine göre piyasa rejimi (düşük/normal/yüksek/panik)."""
    if "VIX" not in macro_df.columns:
        return pd.DataFrame()
    vix = macro_df["VIX"]
    df  = pd.DataFrame(index=macro_df.index)
    df["VIX_level"]      = vix
    df["VIX_chg_5d"]     = vix.pct_change(5)
    df["VIX_regime"]     = pd.cut(vix, bins=[0, 15, 20, 30, 1000],
                                  labels=[0, 1, 2, 3]).astype(float)
    df["VIX_zscore_60d"] = (vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-9)
    return df


def dollar_trend(macro_df: pd.DataFrame) -> pd.DataFrame:
    """USD trendi: kısa ve uzun vadeli momentum."""
    if "UUP" not in macro_df.columns:
        return pd.DataFrame()
    usd = macro_df["UUP"]
    df  = pd.DataFrame(index=macro_df.index)
    df["USD_ret_20d"]    = usd.pct_change(20)
    df["USD_zscore_60d"] = (usd - usd.rolling(60).mean()) / (usd.rolling(60).std() + 1e-9)
    return df


def risk_on_off(macro_df: pd.DataFrame) -> pd.Series:
    """
    Risk iştahı göstergesi: SPY / TLT oranının eğimi.
    Yükselen oran → risk-on; düşen → risk-off.
    """
    if "SPY" in macro_df.columns and "TLT" in macro_df.columns:
        ratio = macro_df["SPY"] / macro_df["TLT"]
        return ratio.pct_change(20).rename("RiskOn_signal")
    return pd.Series(dtype=float, name="RiskOn_signal")


def market_breadth(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Piyasa genişliği:
    - 200 günlük MA üzerindeki hisse oranı
    - 52 haftalık zirveden >%20 uzakta olan hisse oranı (bear market %)
    """
    above_200 = (close_df > close_df.rolling(200).mean()).mean(axis=1)
    peak_52w  = close_df.rolling(252).max()
    drawdown  = (close_df - peak_52w) / (peak_52w + 1e-9)
    bear_pct  = (drawdown < -0.20).mean(axis=1)

    return pd.DataFrame({
        "Breadth_200MA":  above_200,
        "Bear_pct":       bear_pct,
    })


def fed_expectations(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fed faiz beklentisi göstergeleri.
    SHY (1-3Y treasury ETF) ani hareketleri Polymarket'teki
    'Fed faiz indirecek mi?' bahislerine yakın bir proxy sağlar:
    SHY düşüşü = kısa vadeli yield artışı = daha az indirim beklentisi = büyüme hisseleri düşer.
    """
    df = pd.DataFrame(index=macro_df.index)

    # SHY: Fed beklentisi proxy (ters ilişki: SHY düşerse daha az indirim bekleniyor)
    if "SHY" in macro_df.columns:
        shy = macro_df["SHY"]
        df["FedCut_chg_1d"]  = shy.pct_change(1) * 100      # anlık sinyal
        df["FedCut_chg_5d"]  = shy.pct_change(5) * 100      # haftalık momentum
        df["FedCut_zscore"]  = (
            (shy - shy.rolling(60).mean()) / (shy.rolling(60).std() + 1e-9)
        )

    # Kredi riski iştahı: HYG/LQD oranı (yükselen = risk-on credit)
    if "HYG" in macro_df.columns and "LQD" in macro_df.columns:
        hy_ig = macro_df["HYG"] / macro_df["LQD"]
        df["CreditRisk_ratio"]  = hy_ig.pct_change(20)
        df["CreditRisk_zscore"] = (
            (hy_ig - hy_ig.rolling(60).mean()) / (hy_ig.rolling(60).std() + 1e-9)
        )

    # Enflasyon beklentisi: TIP/TLT oranı (yükselen = enflasyon beklentisi artıyor)
    if "TIP" in macro_df.columns and "TLT" in macro_df.columns:
        infl = macro_df["TIP"] / macro_df["TLT"]
        df["Inflation_expect"] = infl.pct_change(20)

    # Yield eğrisi steepness değişimi (ani değişim = Fed beklenti sıfırlanması)
    if "TNX" in macro_df.columns and "IRX" in macro_df.columns:
        spread_lvl = macro_df["TNX"] - macro_df["IRX"]
        df["YieldCurve_chg_5d"]  = spread_lvl.diff(5)   # ani değişim
        df["YieldCurve_chg_20d"] = spread_lvl.diff(20)  # aylık değişim

    # NASDAQ / TLT: teknoloji hisselerinin faiz hassasiyeti
    if "QQQ" in macro_df.columns and "TLT" in macro_df.columns:
        qqq_tlt = macro_df["QQQ"] / macro_df["TLT"]
        df["Tech_Bond_ratio"]  = qqq_tlt.pct_change(5)
        df["Tech_Bond_zscore"] = (
            (qqq_tlt - qqq_tlt.rolling(60).mean()) / (qqq_tlt.rolling(60).std() + 1e-9)
        )

    return df


def sector_rotation(macro_df: pd.DataFrame) -> pd.Series:
    """
    Tech/Consumer vs Utilities/Staples göreli performansı.
    Negatif = risk-off rotasyonu (tech düşüyor, defansif hisseler kazanıyor)
    — tarifeli dönemlerde güçlü bearish sinyali.
    """
    risk_on  = [c for c in ["XLK", "XLY", "XLC"] if c in macro_df.columns]
    risk_off = [c for c in ["XLU", "XLP", "XLV"] if c in macro_df.columns]
    if not risk_on or not risk_off:
        return pd.Series(dtype=float, name="Rotation_signal")
    on_ret  = macro_df[risk_on].pct_change(20).mean(axis=1)
    off_ret = macro_df[risk_off].pct_change(20).mean(axis=1)
    return (on_ret - off_ret).rename("Rotation_signal")


def sector_vs_spy(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm sektör ETF'leri + risk göstergelerinin SPY'a göre 20 günlük relatif getirisi.
    XLB (Materials), XLI (Industrials), XLRE (Real Estate), XLC (Comm.) eklendi.
    """
    df = pd.DataFrame(index=macro_df.index)
    if "SPY" not in macro_df.columns:
        return df
    spy_ret20 = macro_df["SPY"].pct_change(20)
    spy_ret5  = macro_df["SPY"].pct_change(5)

    # Tüm 11 sektör ETF
    for etf in ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB", "XLRE", "XLC"]:
        if etf in macro_df.columns:
            df[f"{etf}_vs_SPY"] = macro_df[etf].pct_change(20) - spy_ret20

    # Risk göstergeleri
    if "GLD" in macro_df.columns:
        df["GLD_vs_SPY"] = macro_df["GLD"].pct_change(20) - spy_ret20
    if "IWM" in macro_df.columns:
        df["IWM_vs_SPY"] = macro_df["IWM"].pct_change(20) - spy_ret20

    # Petrol fiyatı — enerji sektörü için kritik
    if "USO" in macro_df.columns:
        uso = macro_df["USO"]
        df["Oil_ret_20d"]   = uso.pct_change(20).clip(-0.40, 0.40)
        df["Oil_ret_5d"]    = uso.pct_change(5).clip(-0.20, 0.20)
        df["Oil_vs_SPY"]    = uso.pct_change(20) - spy_ret20
        df["Oil_zscore_60d"] = (
            (uso - uso.rolling(60).mean()) / (uso.rolling(60).std() + 1e-9)
        ).clip(-3, 3)

    return df


def post_shock_regime(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-shock toparlanma rejimi sinyalleri.
    Sorun: Nisan 2026 tarife şoku sonrası hisseler MA50/200 altında kaldı
    → model bear tahmin yaptı ama piyasa V-şekilli toparlandı.
    Bu feature set bu rejimi tanımlaması için eklendi.
    """
    df = pd.DataFrame(index=macro_df.index)
    if "VIX" not in macro_df.columns or "SPY" not in macro_df.columns:
        return df

    vix = macro_df["VIX"]
    spy = macro_df["SPY"]

    # VIX spike recovery — 20 ve 40 günlük pencere
    # >1.5 = yakın zamanda spike, şimdi normalleşiyor = recovery modu
    vix_max_20d = vix.rolling(20, min_periods=5).max()
    vix_max_40d = vix.rolling(40, min_periods=10).max()
    df["VIX_spike_ratio"]    = (vix_max_20d / (vix + 1e-9)).clip(1.0, 5.0)
    df["VIX_spike_ratio_40d"] = (vix_max_40d / (vix + 1e-9)).clip(1.0, 5.0)

    # SPY kısa vadeli momentum (toparlanma hızı)
    df["SPY_ret_5d"]  = spy.pct_change(5).clip(-0.15, 0.15)
    df["SPY_ret_10d"] = spy.pct_change(10).clip(-0.20, 0.20)
    df["SPY_ret_21d"] = spy.pct_change(21).clip(-0.30, 0.30)

    # SPY 60 günlük getiri (daha uzun vadeli trend)
    df["SPY_ret_60d"] = spy.pct_change(60).clip(-0.40, 0.40)

    # Post-shock recovery flag (20-gün): VIX >25 gördü, şimdi <22, SPY toparlanıyor
    vix_was_high_20 = (vix_max_20d > 25)
    vix_calm_now    = (vix < 22)
    spy_recovering  = (spy.pct_change(10) > 0.02)
    df["PostShock_recovery"] = (vix_was_high_20 & vix_calm_now & spy_recovering).astype(float)

    # Post-shock recovery flag (40-gün): daha uzun hafıza, SPY koşulsuz
    # Son 40 günde VIX>30 gördü VE şu an <22 = toparlanma rejimi
    vix_was_high_40 = (vix_max_40d > 30)
    df["PostShock_recovery_40d"] = (vix_was_high_40 & vix_calm_now).astype(float)

    # SPY volatilite rejimi: 20 günlük std (normalize)
    spy_std = spy.pct_change().rolling(20).std()
    df["SPY_vol_regime"] = (spy_std * 100).clip(0, 5)

    return df


def government_spending(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Devlet harcaması / savunma sektörü sinyali.
    XAR (Aerospace & Defense ETF): outperform = defense contracts artıyor.
    ITB (Home Construction): outperform = infrastructure harcaması artıyor.
    """
    df = pd.DataFrame(index=macro_df.index)
    if "SPY" not in macro_df.columns:
        return df
    spy_ret20 = macro_df["SPY"].pct_change(20)
    spy_ret60 = macro_df["SPY"].pct_change(60)

    if "XAR" in macro_df.columns:
        df["Defense_vs_SPY_20d"] = macro_df["XAR"].pct_change(20) - spy_ret20
        df["Defense_vs_SPY_60d"] = macro_df["XAR"].pct_change(60) - spy_ret60
        xar = macro_df["XAR"]
        df["Defense_zscore"] = (
            (xar - xar.rolling(60).mean()) / (xar.rolling(60).std() + 1e-9)
        )
    if "ITB" in macro_df.columns:
        df["Infra_vs_SPY_20d"] = macro_df["ITB"].pct_change(20) - spy_ret20
    return df


def options_market_sentiment(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Makro seviyede options piyasası sentiment göstergeleri.
    Hisse başına tarihi options data ücretsiz yok; bunlar piyasa geneli
    ve zaman serisi olduğundan eğitimde kullanılabilir (contamination yok).

    VVIX  = VIX'in VIX'i — options piyasasının ne kadar agresif hedge yaptığı
    VIX3M = 3 aylık VIX — orta vadeli belirsizlik beklentisi
    SKEW  = CBOE SKEW — kuyruk riski / derin OTM put için ödenen prim
    Term Structure = VIX3M / VIX oranı:
        >1 (contango)    = normal, VIX düşecek beklentisi
        <1 (backwardation) = panik, kısa vadeli korku > uzun vadeli
    """
    df = pd.DataFrame(index=macro_df.index)

    vix = macro_df.get("VIX", macro_df.get("^VIX", None))
    if vix is None and "VIX" in macro_df.columns:
        vix = macro_df["VIX"]

    # VVIX: VIX'in kendi volatilitesi — çok yüksek = panik alımı
    if "VVIX" in macro_df.columns:
        vvix = macro_df["VVIX"]
        df["VVIX_level"]      = vvix
        df["VVIX_zscore_60d"] = (
            (vvix - vvix.rolling(60).mean()) / (vvix.rolling(60).std() + 1e-9)
        ).clip(-3, 3)
        df["VVIX_chg_5d"]     = vvix.pct_change(5).clip(-0.30, 0.30)

    # VIX3M: 3 aylık VIX — kısa/orta vade belirsizlik farkı
    if "VIX3M" in macro_df.columns:
        vix3m = macro_df["VIX3M"]
        df["VIX3M_level"] = vix3m
        if vix is not None:
            # Term structure: >1 = contango (normal), <1 = backwardation (panik)
            term_struct = vix3m / (vix + 1e-9)
            df["VIX_term_structure"]   = term_struct.clip(0.5, 2.0)
            df["VIX_term_chg_5d"]      = term_struct.pct_change(5).clip(-0.20, 0.20)
            df["VIX_backwardation"]    = (term_struct < 1.0).astype(float)

    # SKEW: yüksek = derin OTM put prim ödeniyor = büyük düşüş riski fiyatlanıyor
    if "SKEW" in macro_df.columns:
        skew = macro_df["SKEW"]
        df["SKEW_level"]      = skew
        df["SKEW_zscore_60d"] = (
            (skew - skew.rolling(60).mean()) / (skew.rolling(60).std() + 1e-9)
        ).clip(-3, 3)
        df["SKEW_chg_5d"]     = skew.diff(5).clip(-20, 20)
        # Yüksek SKEW + yüksek VIX = maksimum korku kombinasyonu
        if vix is not None:
            df["SKEW_VIX_combo"] = (
                (skew > skew.rolling(60).quantile(0.75)) &
                (vix > 25)
            ).astype(float)

    return df


def build_macro_features(macro_df: pd.DataFrame,
                         close_df: pd.DataFrame) -> pd.DataFrame:
    """Tüm makro özellikleri birleştirir."""
    parts = [
        vix_regime(macro_df),
        dollar_trend(macro_df),
        fed_expectations(macro_df),
    ]

    spread = yield_spread(macro_df)
    if not spread.empty:
        parts.append(spread.to_frame())

    riskOn = risk_on_off(macro_df)
    if not riskOn.empty:
        parts.append(riskOn.to_frame())

    breadth = market_breadth(close_df)
    parts.append(breadth)

    # Sektör rotasyon sinyalleri
    rot = sector_rotation(macro_df)
    if not rot.empty:
        parts.append(rot.to_frame())

    sec_vs_spy = sector_vs_spy(macro_df)
    if not sec_vs_spy.empty:
        parts.append(sec_vs_spy)

    # Devlet harcaması / savunma sinyali
    gov = government_spending(macro_df)
    if not gov.empty:
        parts.append(gov)

    # Post-shock toparlanma rejim sinyalleri
    psr = post_shock_regime(macro_df)
    if not psr.empty:
        parts.append(psr)

    # Makro options sentiment (VVIX, VIX3M, SKEW)
    opts = options_market_sentiment(macro_df)
    if not opts.empty:
        parts.append(opts)

    result = pd.concat([p for p in parts if not p.empty], axis=1)
    return result
