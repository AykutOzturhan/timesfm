"""
Walk-Forward Backtest Engine
============================
Seçilen hisseler üzerinde geçmiş tarihlere gidip o anın verisinden
tahmin üretir, bugünkü gerçekle karşılaştırır, metrik hesaplar
ve modeli iyileştirmek için bulgular üretir.

Kullanım:
  python backtest.py                        # 15 hisse, 3 tarih
  python backtest.py --symbols AAPL,NVDA,UEC
  python backtest.py --improve              # backtest + model iyileştirme
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
# Force UTF-8 output on Windows to avoid Turkish character encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import yfinance as yf
import timesfm
from timesfm import configs
from datetime import datetime, timedelta
import warnings, json, pickle
warnings.filterwarnings("ignore")

from training.xgb_market import MarketXGB
from features.technical import compute_all
from features.macro_features import build_macro_features
from features.sector import _sector_to_etf
from features.news_sentiment import fetch_ticker_sentiment
from features.options_features import fetch_options_snapshot
from sp500.fundamentals import fetch_fundamentals, add_percentile_rank
from config import MACRO_TICKERS, SECTOR_ETFS

# ── Ayarlar ───────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = [
    # ── Teknoloji / Information Technology (50) ─────────────────────────
    "AAPL","MSFT","NVDA","AMD","INTC","AVGO","QCOM","TXN","MU","AMAT",
    "KLAC","LRCX","ADI","MRVL","NXPI","ON","MPWR","SWKS","QRVO","TER",
    "CDNS","SNPS","ANSS","PTC","EPAM","CTSH","IT","INFY","WIT","PLTR",
    "SNOW","DDOG","CRWD","ZS","PANW","FTNT","OKTA","CYBR","S","TENB",
    "ADBE","CRM","ORCL","SAP","NOW","WDAY","VEEV","HUBS","ZM","DOCU",
    # ── İletişim / Communication Services (20) ──────────────────────────
    "META","GOOGL","GOOG","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR",
    "EA","TTWO","RBLX","SNAP","PINS","SPOT","WBD","PARA","FOX","FOXA",
    # ── Tüketici Döngüsel / Consumer Discretionary (30) ─────────────────
    "AMZN","TSLA","HD","MCD","SBUX","NKE","BKNG","MAR","HLT","YUM",
    "CMG","DRI","LVS","WYNN","MGM","DKNG","ROST","TJX","ULTA","LULU",
    "GM","F","RIVN","LCID","POOL","ANF","RL","PVH","VFC","HBI",
    # ── Tüketici Savunma / Consumer Staples (25) ────────────────────────
    "WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","COTY",
    "CLX","CHD","KR","SFM","HELE","EL","REYN","NWL","CAG","CPB",
    "GIS","K","SJM","HRL","MKC",
    # ── Sağlık / Health Care (40) ───────────────────────────────────────
    "JNJ","UNH","ABBV","LLY","PFE","MRK","BMY","AMGN","GILD","REGN",
    "VRTX","BIIB","MRNA","BNTX","INCY","ALNY","SGEN","EXAS","ILMN","DXCM",
    "MDT","ABT","SYK","BSX","EW","ISRG","BDX","ZBH","HOLX","PODD",
    "CI","CVS","HUM","MCK","ABC","CAH","MOH","ELV","CNC","HCA",
    # ── Finans / Financials (30) ────────────────────────────────────────
    "JPM","BAC","WFC","C","GS","MS","BLK","SCHW","CB","AIG",
    "PRU","MET","AFL","ALL","TRV","PGR","AON","MMC","WTW","AJG",
    "V","MA","AXP","DFS","COF","SYF","ALLY","MTB","USB","TFC",
    # ── Enerji / Energy (25) ────────────────────────────────────────────
    "XOM","CVX","COP","EOG","SLB","HAL","BKR","MPC","VLO","PSX",
    "PXD","DVN","FANG","APA","OXY","OVV","MRO","RRC","AR","EQT",
    "LNG","CTRA","SM","UEC","CCJ",
    # ── Sanayi / Industrials (40) — savunma dahil ────────────────────────
    "LMT","RTX","NOC","BA","GD","LHX","LDOS","SAIC","BAH","CACI",
    "CAT","DE","ETN","EMR","ROK","PH","IR","XYL","WM","RSG",
    "UPS","FDX","GE","HON","MMM","ITW","AME","ROP","IEX","FBHS",
    "CARR","OTIS","JCI","TT","LII","GNRC","GXO","CHRW","EXPD","TRMB",
    # ── Malzeme / Materials (20) ────────────────────────────────────────
    "LIN","APD","SHW","ECL","NEM","FCX","CTVA","FMC","DD","DOW",
    "CE","EMN","ALB","MOS","CF","NUE","STLD","CMC","RS","AA",
    # ── Kamu / Utilities (20) ───────────────────────────────────────────
    "NEE","DUK","SO","AEP","EXC","SRE","PCG","ED","WEC","ES",
    "ETR","AEE","LNT","PEG","XEL","CMS","DTE","NI","EVRG","OGE",
    # ── Gayrimenkul / Real Estate (10) ──────────────────────────────────
    "AMT","PLD","EQIX","CCI","SPG","O","PSA","EXR","AVB","EQR",
]

# Tekrarlananları kaldır ama sırayı koru
_seen = set()
DEFAULT_SYMBOLS = [s for s in DEFAULT_SYMBOLS if s not in _seen and not _seen.add(s)]

SECTOR_MAP = {
    # Information Technology
    **{s:"Information Technology" for s in [
        "AAPL","MSFT","NVDA","AMD","INTC","AVGO","QCOM","TXN","MU","AMAT",
        "KLAC","LRCX","ADI","MRVL","NXPI","ON","MPWR","SWKS","QRVO","TER",
        "CDNS","SNPS","ANSS","PTC","EPAM","CTSH","IT","INFY","WIT","PLTR",
        "SNOW","DDOG","CRWD","ZS","PANW","FTNT","OKTA","CYBR","S","TENB",
        "ADBE","CRM","ORCL","SAP","NOW","WDAY","VEEV","HUBS","ZM","DOCU",
    ]},
    # Communication Services
    **{s:"Communication Services" for s in [
        "META","GOOGL","GOOG","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR",
        "EA","TTWO","RBLX","SNAP","PINS","SPOT","WBD","PARA","FOX","FOXA",
    ]},
    # Consumer Discretionary
    **{s:"Consumer Discretionary" for s in [
        "AMZN","TSLA","HD","MCD","SBUX","NKE","BKNG","MAR","HLT","YUM",
        "CMG","DRI","LVS","WYNN","MGM","DKNG","ROST","TJX","ULTA","LULU",
        "GM","F","RIVN","LCID","POOL","ANF","RL","PVH","VFC","HBI",
    ]},
    # Consumer Staples
    **{s:"Consumer Staples" for s in [
        "WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","COTY",
        "CLX","CHD","KR","SFM","HELE","EL","REYN","NWL","CAG","CPB",
        "GIS","K","SJM","HRL","MKC",
    ]},
    # Health Care
    **{s:"Health Care" for s in [
        "JNJ","UNH","ABBV","LLY","PFE","MRK","BMY","AMGN","GILD","REGN",
        "VRTX","BIIB","MRNA","BNTX","INCY","ALNY","SGEN","EXAS","ILMN","DXCM",
        "MDT","ABT","SYK","BSX","EW","ISRG","BDX","ZBH","HOLX","PODD",
        "CI","CVS","HUM","MCK","ABC","CAH","MOH","ELV","CNC","HCA",
    ]},
    # Financials
    **{s:"Financials" for s in [
        "JPM","BAC","WFC","C","GS","MS","BLK","SCHW","CB","AIG",
        "PRU","MET","AFL","ALL","TRV","PGR","AON","MMC","WTW","AJG",
        "V","MA","AXP","DFS","COF","SYF","ALLY","MTB","USB","TFC",
    ]},
    # Energy
    **{s:"Energy" for s in [
        "XOM","CVX","COP","EOG","SLB","HAL","BKR","MPC","VLO","PSX",
        "PXD","DVN","FANG","APA","OXY","OVV","MRO","RRC","AR","EQT",
        "LNG","CTRA","SM","UEC","CCJ",
    ]},
    # Industrials
    **{s:"Industrials" for s in [
        "LMT","RTX","NOC","BA","GD","LDOS","SAIC","BAH","CACI",
        "CAT","DE","ETN","EMR","ROK","PH","IR","XYL","WM","RSG",
        "UPS","FDX","GE","HON","MMM","ITW","AME","ROP","IEX","FBHS",
        "CARR","OTIS","JCI","TT","LII","GNRC","GXO","CHRW","EXPD","TRMB",
    ]},
    # Materials
    **{s:"Materials" for s in [
        "LIN","APD","SHW","ECL","NEM","FCX","CTVA","FMC","DD","DOW",
        "CE","EMN","ALB","MOS","CF","NUE","STLD","CMC","RS","AA",
    ]},
    # Utilities
    **{s:"Utilities" for s in [
        "NEE","DUK","SO","AEP","EXC","SRE","PCG","ED","WEC","ES",
        "ETR","AEE","LNT","PEG","XEL","CMS","DTE","NI","EVRG","OGE",
    ]},
    # Real Estate
    **{s:"Real Estate" for s in [
        "AMT","PLD","EQIX","CCI","SPG","O","PSA","EXR","AVB","EQR",
    ]},
}

# Government contractor flag — savunma ETF (XAR) sinyali bu hisseler için önemli
DEFENSE_CONTRACTORS = {"LMT","RTX","NOC","BA","GD","LDOS","SAIC","BAH","CACI"}
INFRASTRUCTURE_STOCKS = {"CAT","DE","ETN","HON","GE","EMR","XYL"}

# Sabit referans tarihi — her gün değişmesin, karşılaştırılabilir sonuç için
TODAY        = datetime(2026, 6, 18).date()
TEST_OFFSETS = [21, 42, 63, 84, 105]          # 1-5M: temel set
TEST_OFFSETS_EXTENDED = [21, 42, 63, 84, 105,  # 1-5M
                         126, 168, 252]         # 6M, 8M, 12M — normal piyasa dönemleri
HORIZON      = 21             # o noktadan 21 gün ileriye tahmin
CONTEXT      = 252

# Piyasa olayları sözlüğü — her cutoff tarihinde ne olduğunu açıklar
MARKET_EVENTS = {
    "2026-05-20": "Post-tariff recovery (+5.3% SPY 21d)",
    "2026-04-21": "Liberation Day aftermath (VIX spike 60+, recovery başladı)",
    "2026-03-23": "Tariff escalation (Canada/Mexico/China)",
    "2026-02-20": "Normal piyasa (VIX=19)",
    "2026-01-22": "Trump inaugurasyonu sonrası bull (VIX=15.6)",
    "2025-12-11": "Pre-tariff bull market (Aralık rallisi)",
    "2025-10-22": "Normal bull market (Ekim yüksek)",
    "2025-07-22": "Normal bull market (yaz dönemi)",
}

# TÜM makro + sektör ETF sembollerini dahil et — kritik bug düzeltmesi
# Önceden sadece 6 sembol vardı; sektör rotasyon özellikleri (XLY_vs_SPY vb.)
# backtest'te hep 0 oluyordu, oysa bunlar eğitimin top featureları.
_ALL_MACRO_TICKERS = list(MACRO_TICKERS.keys()) + list(SECTOR_ETFS.keys())
MACRO_SYMS = list(dict.fromkeys(_ALL_MACRO_TICKERS))  # deduplicate

BG,PANEL     = "#0d1117","#161b22"
BULL_C       = "#3fb950"
BEAR_C       = "#f85149"
GRID_C       = "#21262d"
TEXT_C       = "#e6edf3"
MUTED_C      = "#8b949e"
CORR_C       = "#f0883e"
HIST_C       = "#58a6ff"

# ── Model Yükleme ─────────────────────────────────────────────────────────
print("TimesFM yukleniyor...")
TFM = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch", device="cpu"
)
TFM.compile(configs.ForecastConfig(
    max_context=512, max_horizon=HORIZON, per_core_batch_size=8
))
print("Model hazir.\n")

try:
    XGB = MarketXGB.load()
    print("XGB modeli yuklendi.")
except FileNotFoundError:
    XGB = None
    print("XGB modeli yok (once train_pipeline.py calistirin)")

# Fundamental percentile cache (inference için)
try:
    _fund_df  = fetch_fundamentals([], force=False)
    FUND_PCT  = add_percentile_rank(_fund_df)
    print(f"Fundamental percentile: {len(FUND_PCT)} hisse.")
except Exception:
    FUND_PCT = None


# ── Yardımcı: Belirtilen tarihe kadar veri + tahmin ──────────────────────
def _datestr(ts) -> str:
    """pd.Timestamp veya datetime'ı YYYY-MM-DD stringe çevir."""
    if hasattr(ts, "date"):
        return str(ts.date())
    return str(ts)[:10]


def fetch_up_to(symbol: str, cutoff: pd.Timestamp) -> pd.Series:
    """Verilen tarihe kadar (dahil) kapanış fiyatı."""
    end_str = _datestr(cutoff + timedelta(days=1))
    raw = yf.download(symbol, start="2022-01-01", end=end_str,
                      interval="1d", auto_adjust=True, progress=False)
    close = raw["Close"].squeeze().dropna()
    return close[close.index <= pd.Timestamp(cutoff)]


def fetch_macro_up_to(cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Tüm makro + sektör ETF verilerini indirir (önceden yalnızca 6 sembol vardı —
    bu yüzden sector rotation, GLD/IWM vs SPY, fed expectations özellikleri
    backtest sırasında hep 0 çıkıyordu).
    """
    end_str = _datestr(cutoff + timedelta(days=1))
    # Batch halinde indir (yfinance çok fazla sembol için daha stabil)
    frames = {}
    batch_size = 20
    for i in range(0, len(MACRO_SYMS), batch_size):
        batch = MACRO_SYMS[i:i+batch_size]
        try:
            raw = yf.download(batch, start="2022-01-01", end=end_str,
                              auto_adjust=True, progress=False)
            if raw.empty:
                continue
            for sym in batch:
                col = sym.replace("^","")
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        lvl0 = raw.columns.get_level_values(0)
                        lvl1 = raw.columns.get_level_values(1)
                        if "Close" in lvl0 and sym in lvl1:
                            s = raw["Close"][sym].dropna()
                        elif sym in lvl0 and "Close" in raw[sym].columns:
                            s = raw[sym]["Close"].dropna()
                        else:
                            continue
                    else:
                        s = raw["Close"].dropna() if len(batch) == 1 else raw["Close"][sym].dropna()
                    frames[col] = s[s.index <= pd.Timestamp(cutoff)]
                except Exception:
                    pass
        except Exception as e:
            print(f"  Makro batch hatası ({batch[0]}..): {e}")
    return pd.DataFrame(frames)


def build_feat_vec(symbol: str, close: pd.Series,
                   macro_df: pd.DataFrame) -> np.ndarray:
    """Son tarihteki özellik vektörü."""
    try:
        close_df  = close.to_frame(symbol)
        high_df   = close_df * 1.005
        low_df    = close_df * 0.995
        vol_df    = close_df * 1000   # hacim yok, proxy
        tech = compute_all(close_df, high_df, low_df, vol_df)
        parts = []
        for name, df in tech.items():
            if symbol in df.columns:
                parts.append(df[symbol].rename(name))
        macro_feats = build_macro_features(macro_df, close_df)
        # market_breadth() tek hisse için anlamsız (0/1) — SPY ile tahmin et
        if "SPY" in macro_df.columns:
            spy = macro_df["SPY"].dropna()
            if len(spy) >= 200:
                spy_ma200 = spy.rolling(200).mean()
                spy_ratio = (spy / (spy_ma200 + 1e-9) - 1).clip(-0.30, 0.30)
                # -0.30 -> 0.0, 0 -> 0.5, +0.30 -> 1.0
                macro_feats["Breadth_200MA"] = (0.5 + spy_ratio / 0.30 * 0.5).reindex(
                    macro_feats.index).ffill().fillna(0.5)
                spy_peak52 = spy.rolling(252).max()
                spy_dd     = (spy - spy_peak52) / (spy_peak52 + 1e-9)
                macro_feats["Bear_pct"] = (spy_dd < -0.10).astype(float).reindex(
                    macro_feats.index).ffill().fillna(0.0)
        parts.append(macro_feats)
        etf = _sector_to_etf(SECTOR_MAP.get(symbol,""))
        etf_col = etf.replace("^","") if etf else None
        if etf_col and etf_col in macro_df.columns:
            etf_ret20 = macro_df[etf_col].pct_change(20)
            stk_ret20 = close.pct_change(20)
            rs = (stk_ret20 - etf_ret20).rename("RS_sector_20d")
            parts.append(rs)

        # Government/Defense özel sinyali
        if symbol in DEFENSE_CONTRACTORS and "XAR" in macro_df.columns and "SPY" in macro_df.columns:
            defense_sig = (macro_df["XAR"].pct_change(20) - macro_df["SPY"].pct_change(20)).rename("Defense_sector_edge")
            parts.append(defense_sig)
        if symbol in INFRASTRUCTURE_STOCKS and "ITB" in macro_df.columns and "SPY" in macro_df.columns:
            infra_sig = (macro_df["ITB"].pct_change(20) - macro_df["SPY"].pct_change(20)).rename("Infra_sector_edge")
            parts.append(infra_sig)

        # Fundamental percentile özellikleri — sabit değerler, tüm tarihlere yay
        if FUND_PCT is not None and symbol in FUND_PCT.index:
            for k, v in FUND_PCT.loc[symbol].to_dict().items():
                if not pd.isna(v):
                    parts.append(pd.Series(float(v), index=close.index, name=k))

        feat_df = pd.concat([p.to_frame() if isinstance(p, pd.Series) else p
                             for p in parts], axis=1)
        feat_df = feat_df.ffill().fillna(0.0)

        # Haber sentiment (son 30 gün)
        try:
            sent = fetch_ticker_sentiment(symbol)
            if not sent.empty:
                sent = sent.reindex(feat_df.index).ffill(limit=5).fillna(0.0)
                feat_df["sent_raw"]   = sent.values
                feat_df["sent_ma5"]   = sent.rolling(5,  min_periods=1).mean().values
                feat_df["sent_trend"] = (sent.rolling(5, min_periods=1).mean()
                                         - sent.rolling(20, min_periods=1).mean()).values
        except Exception:
            feat_df["sent_raw"]   = 0.0
            feat_df["sent_ma5"]   = 0.0
            feat_df["sent_trend"] = 0.0

        # Options IV + hacim snapshot (bugünkü anlık veri — training ile tutarlı)
        try:
            opts = fetch_options_snapshot(symbol)
            def _safe(v, default=0.0):
                if v is None: return default
                try:
                    f = float(v)
                    return default if (np.isnan(f) or np.isinf(f)) else f
                except (TypeError, ValueError):
                    return default
            atm_call = _safe(opts.get("atm_iv_call"))
            atm_put  = _safe(opts.get("atm_iv_put"))
            feat_df["opt_atm_iv"]       = (atm_call + atm_put) / 2 if (atm_call > 0 or atm_put > 0) else 0.0
            feat_df["opt_put_call"]     = _safe(opts.get("put_call_ratio"))
            feat_df["opt_put_call_vol"] = _safe(opts.get("put_call_vol_ratio"))
            feat_df["opt_iv_skew"]      = _safe(opts.get("iv_skew"))
            avg_call = _safe(opts.get("avg_call_iv"))
            avg_put  = _safe(opts.get("avg_put_iv"))
            feat_df["opt_avg_iv"]        = (avg_call + avg_put) / 2 if (avg_call > 0 or avg_put > 0) else 0.0
            feat_df["opt_iv_percentile"] = _safe(opts.get("iv_percentile"))
            feat_df["opt_call_spike"]    = _safe(opts.get("call_vol_spike"))
            feat_df["opt_put_spike"]     = _safe(opts.get("put_vol_spike"))
        except Exception:
            feat_df["opt_atm_iv"]       = 0.0
            feat_df["opt_put_call"]     = 0.0
            feat_df["opt_put_call_vol"] = 0.0
            feat_df["opt_iv_skew"]      = 0.0
            feat_df["opt_avg_iv"]       = 0.0
            feat_df["opt_iv_percentile"]= 0.0
            feat_df["opt_call_spike"]   = 0.0
            feat_df["opt_put_spike"]    = 0.0

        feat_df = feat_df.replace([np.inf, -np.inf], 0.0)
        last = feat_df.iloc[-1]
        return last.values.astype(np.float32), list(last.index)
    except Exception:
        return np.zeros(30, dtype=np.float32), []


def run_single(symbol: str, cutoff: pd.Timestamp,
               macro_df: pd.DataFrame | None = None) -> dict | None:
    """
    cutoff anındaki veriden 21 günlük tahmin üret,
    gerçek fiyatla karşılaştır.
    macro_df önceden çekilmişse tekrar indirilmez (büyük hız kazancı).
    """
    try:
        close    = fetch_up_to(symbol, cutoff)
        if len(close) < CONTEXT + 5:
            return None
        ctx      = close.values.astype(np.float32)[-CONTEXT:]
        price_at_cutoff = float(ctx[-1])

        # TimesFM tahmin
        pts, qnts = TFM.forecast(horizon=HORIZON, inputs=[ctx])
        fc_base   = pts[0]   # (21,)

        # XGB düzeltme
        correction = 0.0
        if macro_df is None:
            macro_df = fetch_macro_up_to(cutoff)
        feat_vec, feat_names = build_feat_vec(symbol, close, macro_df)
        if XGB is not None and XGB.feature_names:
            aligned = np.zeros(len(XGB.feature_names), dtype=np.float32)
            name2i  = {n: i for i, n in enumerate(feat_names)}
            for i, col in enumerate(XGB.feature_names):
                if col in name2i:
                    aligned[i] = feat_vec[name2i[col]]
            raw_corr  = float(XGB.predict(aligned.reshape(1,-1))[0])
            # Adaptif scale: XGB ve TFM anlaşmıyorsa VE XGB güçlü sinyal veriyorsa
            # daha büyük düzeltme uygula (TimesFM trend extrapolasyonunu geçersiz kıl).
            # Mantık: VIX/tarife krizinde TFM yükselen trendi devam ettiriyor,
            # XGB ise makro bozulmayı görüyor — XGB'ye daha fazla ağırlık ver.
            tfm_net = float(fc_base[-1]) - price_at_cutoff
            xgb_agrees = (raw_corr > 0) == (tfm_net > 0)
            if not xgb_agrees and abs(raw_corr) >= 0.25:
                scale = 0.20   # güçlü anlaşmazlık: XGB'ye yüksek güven
            else:
                scale = 0.04   # anlaşma veya zayıf sinyal: hafif düzeltme
            correction = raw_corr * scale * price_at_cutoff

        fc_corrected = fc_base + correction

        # Gerçek fiyatlar (cutoff'tan sonraki HORIZON iş günü)
        future_start = cutoff + timedelta(days=1)
        future_end   = cutoff + timedelta(days=HORIZON * 2)
        raw_future   = yf.download(symbol,
                                   start=_datestr(future_start),
                                   end=_datestr(future_end),
                                   interval="1d", auto_adjust=True, progress=False)
        actual_close = raw_future["Close"].squeeze().dropna().values[:HORIZON]
        if len(actual_close) < 5:
            return None

        n_overlap = min(len(fc_base), len(actual_close))
        fc_used   = fc_corrected[:n_overlap]
        act_used  = actual_close[:n_overlap].astype(np.float32)

        mae      = float(np.mean(np.abs(fc_used - act_used)))
        mape     = float(np.mean(np.abs(fc_used - act_used) / (np.abs(act_used) + 1e-9))) * 100
        fc_dir   = float(fc_corrected[-1]) - price_at_cutoff
        act_dir  = float(actual_close[-1]) - price_at_cutoff
        dir_ok   = (fc_dir * act_dir) > 0

        # Fiyat t+21 tahmini vs gerçek
        fc_end   = float(fc_corrected[min(20, n_overlap-1)])
        act_end  = float(actual_close[min(20, n_overlap-1)])

        return {
            "symbol":          symbol,
            "cutoff":          str(cutoff.date()),
            "price_at_cutoff": price_at_cutoff,
            "fc_end":          fc_end,
            "act_end":         act_end,
            "fc_change_pct":   (fc_end - price_at_cutoff) / price_at_cutoff * 100,
            "act_change_pct":  (act_end - price_at_cutoff) / price_at_cutoff * 100,
            "mae":             mae,
            "mape":            mape,
            "dir_correct":     dir_ok,
            "correction_usd":  correction,
            "fc_series":       fc_corrected.tolist(),
            "base_series":     fc_base.tolist(),
            "act_series":      actual_close.tolist(),
            "quants":          qnts[0].tolist(),
        }
    except Exception as e:
        print(f"  HATA [{symbol} @ {cutoff.date()}]: {e}")
        return None


# ── Ana Backtest Döngüsü ─────────────────────────────────────────────────
def run_backtest(symbols: list[str], extended: bool = False) -> pd.DataFrame:
    active_offsets = TEST_OFFSETS_EXTENDED if extended else TEST_OFFSETS
    results = []
    total   = len(symbols) * len(active_offsets)
    done    = 0

    # Makro veriyi her benzersiz cutoff için bir kez indir
    print("  Makro veri önbellekleniyor...")
    macro_cache: dict[int, pd.DataFrame] = {}
    for offset in active_offsets:
        cutoff = pd.Timestamp(TODAY) - pd.offsets.BDay(offset)
        event  = MARKET_EVENTS.get(str(cutoff.date()), "")
        print(f"    {cutoff.date()} için makro veri indiriliyor...", end=" ", flush=True)
        macro_cache[offset] = fetch_macro_up_to(cutoff)
        print(f"OK ({len(macro_cache[offset].columns)} kolon)  {event}")

    for sym in symbols:
        for offset in active_offsets:
            cutoff = pd.Timestamp(TODAY) - pd.offsets.BDay(offset)
            months_ago = offset // 21
            print(f"  [{done+1:3d}/{total}] {sym:6s} @ {cutoff.date()} "
                  f"(~{months_ago}ay önce) ...", end=" ", flush=True)
            r = run_single(sym, cutoff, macro_df=macro_cache[offset])
            if r:
                results.append(r)
                sign  = "OK" if r["dir_correct"] else "XX"
                print(f"MAPE={r['mape']:.1f}%  dir={sign}")
            else:
                print("atlandı")
            done += 1

    return pd.DataFrame(results)


# ── Metrik Hesaplama ─────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> dict:
    return {
        "n_tests":        len(df),
        "mean_mape":      df["mape"].mean(),
        "median_mape":    df["mape"].median(),
        "dir_acc":        df["dir_correct"].mean() * 100,
        "mae_mean":       df["mae"].mean(),
        "best_symbol":    df.groupby("symbol")["mape"].mean().idxmin(),
        "worst_symbol":   df.groupby("symbol")["mape"].mean().idxmax(),
        "by_offset":      df.groupby("cutoff")["mape"].mean().to_dict(),
    }


# ── Backtest Dashboard ───────────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame, metrics: dict, out_path: str):
    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle(
        f"Backtest Analizi — {metrics['n_tests']} test  |  "
        f"Ort. MAPE: {metrics['mean_mape']:.1f}%  |  "
        f"Yön Doğruluk: {metrics['dir_acc']:.1f}%",
        fontsize=14, fontweight="bold", color=TEXT_C, y=0.98
    )

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.52, wspace=0.35,
                           left=0.05, right=0.97, top=0.94, bottom=0.05)

    def ax_style(ax, title=""):
        ax.set_facecolor(PANEL)
        ax.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.6)
        for sp in ax.spines.values(): sp.set_color(GRID_C)
        if title: ax.set_title(title, color=TEXT_C, fontsize=10, pad=6)
        ax.tick_params(colors=MUTED_C, labelsize=7.5)

    # ── 1. Hisse bazında MAPE ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax_style(ax1, "Hisse Bazında Ortalama MAPE (%)")
    sym_mape = df.groupby("symbol")["mape"].mean().sort_values()
    clrs = [BULL_C if v < 5 else CORR_C if v < 10 else BEAR_C for v in sym_mape]
    bars = ax1.barh(sym_mape.index, sym_mape.values, color=clrs, alpha=0.85)
    ax1.axvline(5,  color=BULL_C, linewidth=0.8, linestyle=":", alpha=0.6)
    ax1.axvline(10, color=BEAR_C, linewidth=0.8, linestyle=":", alpha=0.6)
    for bar, val in zip(bars, sym_mape.values):
        ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=7.5, color=TEXT_C)
    ax1.set_xlabel("MAPE (%)", color=MUTED_C, fontsize=8)

    # ── 2. Yön doğruluğu pie ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values(): sp.set_color(GRID_C)
    dir_ok  = int(df["dir_correct"].sum())
    dir_no  = len(df) - dir_ok
    ax2.pie([dir_ok, dir_no],
            labels=[f"Doğru\n{dir_ok}", f"Yanlış\n{dir_no}"],
            colors=[BULL_C, BEAR_C], startangle=90,
            textprops={"color": TEXT_C, "fontsize": 9},
            wedgeprops={"edgecolor": BG, "linewidth": 2})
    ax2.set_title(f"Yön Doğruluğu\n{metrics['dir_acc']:.1f}%",
                  color=TEXT_C, fontsize=10, pad=6)

    # ── 3. MAPE dağılımı histogram ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    ax_style(ax3, "MAPE Dağılımı")
    bins = np.linspace(0, df["mape"].quantile(0.95), 20)
    ax3.hist(df["mape"], bins=bins, color=CORR_C, alpha=0.75, edgecolor=BG)
    ax3.axvline(df["mape"].median(), color=BULL_C, linewidth=1.5,
                linestyle="--", label=f"Medyan {df['mape'].median():.1f}%")
    ax3.legend(fontsize=7.5, labelcolor=TEXT_C, facecolor=PANEL, edgecolor=GRID_C)
    ax3.set_xlabel("MAPE (%)", color=MUTED_C, fontsize=8)

    # ── 4. Tahmin vs Gerçek scatter ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax_style(ax4, "Tahmin Edilen vs Gerçek Değişim (%)")
    fc_chg  = df["fc_change_pct"].values
    act_chg = df["act_change_pct"].values
    clrs_s  = [BULL_C if d else BEAR_C for d in df["dir_correct"]]
    ax4.scatter(act_chg, fc_chg, c=clrs_s, alpha=0.7, s=60, edgecolors=BG, linewidths=0.5)
    lim = max(abs(fc_chg).max(), abs(act_chg).max()) * 1.15
    ax4.set_xlim(-lim, lim); ax4.set_ylim(-lim, lim)
    ax4.axhline(0, color=MUTED_C, linewidth=0.5)
    ax4.axvline(0, color=MUTED_C, linewidth=0.5)
    ax4.plot([-lim, lim], [-lim, lim], color=MUTED_C, linewidth=0.8,
             linestyle=":", alpha=0.5, label="Mükemmel tahmin")
    ax4.set_xlabel("Gerçek Değişim (%)", color=MUTED_C, fontsize=8)
    ax4.set_ylabel("Tahmin Değişim (%)", color=MUTED_C, fontsize=8)
    # R² hesapla
    ss_res = np.sum((act_chg - fc_chg)**2)
    ss_tot = np.sum((act_chg - act_chg.mean())**2) + 1e-9
    r2 = max(0, 1 - ss_res / ss_tot)
    ax4.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax4.transAxes,
             color=TEXT_C, fontsize=9, fontweight="bold")
    ax4.legend(fontsize=7.5, labelcolor=TEXT_C, facecolor=PANEL, edgecolor=GRID_C)

    # ── 5. Hisse fiyat serileri (en iyi 4) ───────────────────────────
    best4 = df.groupby("symbol")["mape"].mean().nsmallest(4).index.tolist()
    for i, sym in enumerate(best4):
        col = 2 + (i % 2)
        row = 1 + (i // 2)
        ax  = fig.add_subplot(gs[row, col])
        ax_style(ax, f"{sym}  MAPE={df[df.symbol==sym]['mape'].mean():.1f}%")
        # En son test noktasını al
        sub = df[df.symbol == sym].sort_values("cutoff").iloc[-1]
        n   = min(len(sub["act_series"]), len(sub["fc_series"]))
        t   = np.arange(n)
        ax.plot(t, sub["act_series"][:n], color=HIST_C, linewidth=1.5,
                label="Gerçek")
        ax.plot(t, sub["fc_series"][:n], color=CORR_C, linewidth=1.5,
                linestyle="--", label="Tahmin")
        ax.plot(t, sub["base_series"][:n], color=MUTED_C, linewidth=1,
                linestyle=":", alpha=0.6, label="Base")
        ax.set_xlabel("Gün", color=MUTED_C, fontsize=7)
        ax.legend(fontsize=6.5, labelcolor=TEXT_C, facecolor=PANEL,
                  edgecolor=GRID_C, framealpha=0.5)

    # ── 6. Zaman bazında MAPE (kaç ay önce?) ────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    ax_style(ax6, "Horizon Bazında Ortalama MAPE (1M, 2M, 3M önce)")
    df["offset_label"] = df.apply(
        lambda r: f"~{TEST_OFFSETS.index(min(TEST_OFFSETS, key=lambda x: abs(x - (pd.Timestamp(TODAY)-pd.Timestamp(r['cutoff'])).days)))//21}M önce",
        axis=1
    )
    off_mape = df.groupby("offset_label")["mape"].mean().sort_index()
    ax6.bar(off_mape.index, off_mape.values,
            color=[BULL_C, CORR_C, BEAR_C][:len(off_mape)], alpha=0.8)
    ax6.set_ylabel("Ort. MAPE (%)", color=MUTED_C, fontsize=8)
    for x, v in zip(off_mape.index, off_mape.values):
        ax6.text(x, v + 0.1, f"{v:.1f}%", ha="center",
                 fontsize=9, color=TEXT_C, fontweight="bold")

    # ── 7. Özet metrik tablosu ───────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.set_facecolor(PANEL)
    ax7.axis("off")
    ax7.set_title("Özet Metrikler", color=TEXT_C, fontsize=10, pad=6)
    rows_tbl = [
        ("Test sayısı",       str(metrics["n_tests"]),       TEXT_C),
        ("Ort. MAPE",         f"{metrics['mean_mape']:.2f}%", CORR_C),
        ("Medyan MAPE",       f"{metrics['median_mape']:.2f}%", CORR_C),
        ("Yön Doğruluk",      f"{metrics['dir_acc']:.1f}%",
         BULL_C if metrics["dir_acc"] > 55 else BEAR_C),
        ("Ort. MAE",          f"${metrics['mae_mean']:.2f}",  TEXT_C),
        ("En İyi Hisse",      metrics["best_symbol"],         BULL_C),
        ("En Zor Hisse",      metrics["worst_symbol"],        BEAR_C),
    ]
    y0 = 0.92
    for label, val, clr in rows_tbl:
        ax7.text(0.04, y0, label, transform=ax7.transAxes,
                 fontsize=9, color=MUTED_C, va="top")
        ax7.text(0.62, y0, val,   transform=ax7.transAxes,
                 fontsize=9, color=clr, va="top", fontweight="bold")
        y0 -= 0.13

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nDashboard kaydedildi: {Path(out_path).name}")


# ── Model İyileştirme ────────────────────────────────────────────────────
IMPROVEMENT_LOG = Path("c:/Users/aykut/OneDrive/Masaüstü/trade/timesfm/improvement_log.json")


def _load_improvement_log() -> list:
    if IMPROVEMENT_LOG.exists():
        import json
        return json.loads(IMPROVEMENT_LOG.read_text())
    return []


def _save_improvement_log(log: list):
    import json
    IMPROVEMENT_LOG.write_text(json.dumps(log, indent=2))


def improve_model(df: pd.DataFrame):
    """
    Backtest bulgularına göre XGBoost modelini yeniden eğitir.

    Stratejiler:
    1. Eski modelin scaler'ını koru  — yeni scaler inference tutarsızlığına yol açıyor
    2. Proper train/val split        — in-sample eval overfitting'i gizler
    3. Recency weighting             — son pencereler daha fazla önem taşır
    4. Early stopping                — overfitting'i önler (eval_set ile)
    5. Geniş grid                    — 6 farklı config, val-dir doğruluğu optimize
    6. Sadece gerçekten iyi ise kaydet — validation kötüleşirse eski modeli koru
    7. İterasyon geçmişi             — her çalışmada loga yazar
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import RobustScaler
    import xgboost as xgb
    from config import DATA_DIR
    import json

    print("\n[MODELİ İYİLEŞTİR — Akıllı Versiyon]")

    dataset_path = DATA_DIR / "training_windows.npz"
    if not dataset_path.exists():
        print("  Eğitim verisi bulunamadı.")
        return

    data   = np.load(dataset_path, allow_pickle=True)
    X_feat = data["X_feat"]
    y      = data["y"]
    n      = len(y)
    if "feat_cols" in data:
        true_feat_names = list(data["feat_cols"])
    else:
        true_feat_names = None

    if n < 50:
        print("  Yetersiz veri.")
        return

    print(f"  Eğitim seti: {n:,} örnek, {X_feat.shape[1]} özellik")

    # ── 1. Backtest'ten MAPE-bazlı ağırlıklar ─────────────────────────
    sym_mape  = df.groupby("symbol")["mape"].mean()
    sym_dirok = df.groupby("symbol")["dir_correct"].mean()

    high_mape = sym_mape[sym_mape > sym_mape.quantile(0.75)].index.tolist()
    low_mape  = sym_mape[sym_mape < sym_mape.quantile(0.25)].index.tolist()
    bad_dir   = sym_dirok[sym_dirok < 0.4].index.tolist()

    print(f"  Yüksek hata hisseler: {high_mape}")
    print(f"  Kötü yön tahmini   : {bad_dir}")
    print(f"  İyi hisseler       : {low_mape}")

    # ── 2. Train/Val split (son %15 = validation, zaman sırası korunur) ─
    split          = int(n * 0.85)
    X_tr, y_tr     = X_feat[:split], y[:split]
    X_va, y_va     = X_feat[split:], y[split:]

    # ── 3. Ağırlık vektörü ─────────────────────────────────────────────
    sample_weights        = np.ones(n, dtype=np.float32)
    recency_start         = int(n * 0.85)
    sample_weights[recency_start:] *= 1.3

    # ── 4. Scaler: eski modelin scaler'ını yeniden kullan ─────────────
    # Yeni bir scaler fit etmek inference'ı bozuyor çünkü build_feat_vec()
    # farklı bir dağılımdan özellik üretiyor. Tutarlılık için eski scaler koru.
    try:
        old_model  = MarketXGB.load()
        scaler     = old_model.scaler
        feat_names = true_feat_names if true_feat_names else old_model.feature_names
        old_va_preds = old_model.predict(X_va)
        old_va_dir   = float(np.mean(np.sign(old_va_preds) == np.sign(y_va)))
    except Exception:
        scaler = RobustScaler()
        scaler.fit(X_tr)
        feat_names   = true_feat_names
        old_va_dir   = None

    Xs_tr = scaler.transform(X_tr)
    Xs_va = scaler.transform(X_va)

    if old_va_dir is not None:
        print(f"  Eski model val: dir={old_va_dir*100:.1f}%")

    # ── 5. Hiperparametre grid + early stopping ────────────────────────
    print("\n  Hiperparametre araması (10 config, early stopping)...")
    best_score = float("inf")
    best_cfg   = {}
    results    = []

    # Classifier için binary label
    y_tr_dir = (y_tr > 0).astype(int)
    y_va_dir = (y_va > 0).astype(int)

    configs_to_try = [
        # Mevcut en iyi (depth=8)
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05,
         "subsample": 0.7, "colsample_bytree": 0.6,
         "gamma": 0.4, "min_child_weight": 10,
         "reg_alpha": 0.3, "reg_lambda": 2.0},
        # depth=8 varyasyonları
        {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.06,
         "subsample": 0.7, "colsample_bytree": 0.6,
         "gamma": 0.5, "min_child_weight": 12,
         "reg_alpha": 0.4, "reg_lambda": 2.5},
        {"n_estimators": 350, "max_depth": 8, "learning_rate": 0.04,
         "subsample": 0.65, "colsample_bytree": 0.55,
         "gamma": 0.4, "min_child_weight": 10,
         "reg_alpha": 0.3, "reg_lambda": 2.0},
        # depth=9
        {"n_estimators": 250, "max_depth": 9, "learning_rate": 0.05,
         "subsample": 0.65, "colsample_bytree": 0.55,
         "gamma": 0.5, "min_child_weight": 12,
         "reg_alpha": 0.4, "reg_lambda": 2.5},
        {"n_estimators": 200, "max_depth": 9, "learning_rate": 0.07,
         "subsample": 0.65, "colsample_bytree": 0.5,
         "gamma": 0.6, "min_child_weight": 15,
         "reg_alpha": 0.5, "reg_lambda": 3.0},
        # depth=10
        {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.05,
         "subsample": 0.6, "colsample_bytree": 0.5,
         "gamma": 0.6, "min_child_weight": 15,
         "reg_alpha": 0.5, "reg_lambda": 3.0},
        # depth=7 (önceki iyi)
        {"n_estimators": 350, "max_depth": 7, "learning_rate": 0.04,
         "subsample": 0.75, "colsample_bytree": 0.65,
         "gamma": 0.3, "min_child_weight": 7,
         "reg_alpha": 0.2, "reg_lambda": 1.5},
        # n=450, depth=6 (güçlü alternatif)
        {"n_estimators": 450, "max_depth": 6, "learning_rate": 0.035,
         "subsample": 0.85, "colsample_bytree": 0.8,
         "gamma": 0.15, "min_child_weight": 5,
         "reg_alpha": 0.1, "reg_lambda": 1.2},
        # depth=8, daha az ağaç
        {"n_estimators": 180, "max_depth": 8, "learning_rate": 0.08,
         "subsample": 0.7, "colsample_bytree": 0.6,
         "gamma": 0.5, "min_child_weight": 12,
         "reg_alpha": 0.4, "reg_lambda": 2.0},
        # depth=6 baseline (karşılaştırma için)
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.04,
         "subsample": 0.8, "colsample_bytree": 0.7,
         "gamma": 0.2, "min_child_weight": 5,
         "reg_alpha": 0.1, "reg_lambda": 1.0},
    ]

    best_iterations = {}
    best_ci = 0

    for ci, cfg in enumerate(configs_to_try):
        m = xgb.XGBClassifier(
            **cfg, objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=42,
            early_stopping_rounds=40,
        )
        m.fit(
            Xs_tr, y_tr_dir,
            sample_weight=sample_weights[:split],
            eval_set=[(Xs_va, y_va_dir)],
            verbose=False,
        )
        best_iter = getattr(m, "best_iteration", cfg["n_estimators"] - 1)
        best_iterations[ci] = max(100, best_iter + 1)
        prob_va  = m.predict_proba(Xs_va)[:, 1]
        val_dir  = float(np.mean((prob_va > 0.5) == y_va_dir))
        combo    = 1.0 - val_dir   # lower is better; maximize direction accuracy
        est_str  = f"n={cfg['n_estimators']},d={cfg['max_depth']},lr={cfg['learning_rate']}"
        print(f"    [{est_str}] val_dir={val_dir*100:.1f}%  "
              f"combo={combo:.5f}  best_iter={best_iterations[ci]}")
        results.append({"cfg": cfg, "dir": val_dir,
                        "combo": combo, "best_iter": best_iterations[ci]})
        if combo < best_score:
            best_score = combo
            best_cfg   = cfg
            best_ci    = ci

    actual_trees = best_iterations.get(best_ci, best_cfg["n_estimators"])
    print(f"\n  En iyi config: n={best_cfg['n_estimators']}, "
          f"depth={best_cfg['max_depth']}, lr={best_cfg['learning_rate']}  "
          f"(val_dir={100*(1-best_score):.1f}%, kullanılacak_ağaç={actual_trees})")

    # ── 6. Final model — train veriyle eğit (scaler eski), val ile değerlendir
    final_cfg = dict(best_cfg)
    final_cfg["n_estimators"] = actual_trees

    final_xgb = xgb.XGBClassifier(
        **final_cfg, objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    final_xgb.fit(Xs_tr, y_tr_dir, sample_weight=sample_weights[:split], verbose=False)
    print(f"  Kullanılan ağaç sayısı: {actual_trees}")

    new_model              = MarketXGB()
    new_model.scaler       = scaler
    new_model.feature_names = feat_names
    new_model.model        = final_xgb

    # ── 7. Karşılaştırma — SADECE validation seti üzerinde ────────────
    new_va_preds = new_model.predict(X_va)    # (prob-0.5): sign = direction
    new_va_dir   = float(np.mean(np.sign(new_va_preds) == np.sign(y_va)))

    print(f"\n  {'Metrik (VAL)':<20} {'Eski':>10} {'Yeni':>10} {'delta':>10}")
    print(f"  {'-'*52}")
    if old_va_dir is not None:
        delta_dir = (new_va_dir - old_va_dir) * 100
        print(f"  {'VAL Yön Doğ.':<20} {old_va_dir*100:>9.1f}% {new_va_dir*100:>9.1f}% {delta_dir:>+9.1f}pp")
    else:
        print(f"  {'VAL Yön Doğ.':<20} {'--':>10} {new_va_dir*100:>9.1f}%")

    # ── 8. Sadece gerçekten daha iyiyse kaydet ────────────────────────
    if old_va_dir is not None and new_va_dir <= old_va_dir:
        print(f"\n  Yeni model val'da daha kötü ({new_va_dir*100:.1f}% <= {old_va_dir*100:.1f}%) "
              f"— eski model korunuyor.")
        log = _load_improvement_log()
        log.append({
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_stocks":    int(df["symbol"].nunique()),
            "n_tests":     int(len(df)),
            "backtest_mape": float(df["mape"].mean()),
            "backtest_dir":  float(df["dir_correct"].mean()),
            "val_dir":     float(new_va_dir),
            "saved":       False,
            "best_cfg":    {k: v for k, v in best_cfg.items()},
            "actual_trees": int(actual_trees),
        })
        _save_improvement_log(log)
        return None

    new_model.save()

    # ── 9. İterasyon logu ─────────────────────────────────────────────
    log = _load_improvement_log()
    log.append({
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_stocks":    int(df["symbol"].nunique()),
        "n_tests":     int(len(df)),
        "backtest_mape": float(df["mape"].mean()),
        "backtest_dir":  float(df["dir_correct"].mean()),
        "val_dir":     float(new_va_dir),
        "saved":       True,
        "best_cfg":    {k: v for k, v in best_cfg.items()},
        "actual_trees": int(actual_trees),
    })
    _save_improvement_log(log)
    print(f"\n  İterasyon #{len(log)} loga yazıldı: {IMPROVEMENT_LOG.name}")
    if len(log) > 1:
        prev = log[-2]
        print(f"  Onceki backtest MAPE: {prev['backtest_mape']:.2f}%  ->  "
              f"Simdiki: {log[-1]['backtest_mape']:.2f}%")

    return new_model


# ── Ana Çalışma ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="",
                        help="Virgülle ayrılmış hisse kodları (boşsa default)")
    parser.add_argument("--improve", action="store_true",
                        help="Backtest sonrası modeli iyileştir")
    parser.add_argument("--sample", type=int, default=0,
                        help="Test için rastgele N hisse seç (0=hepsi)")
    parser.add_argument("--quick", action="store_true",
                        help="Hızlı test: 30 hisse, 3 tarih")
    parser.add_argument("--reliable-only", action="store_true",
                        help="Sadece güvenilir evreni kullan (reliable_universe.json)")
    parser.add_argument("--extended", action="store_true",
                        help="Genişletilmiş tarih aralığı: 8 tarih (1M-12M)")
    args = parser.parse_args()

    # Güvenilir evren yükle
    reliable_universe_path = Path(__file__).parent / "data" / "reliable_universe.json"
    reliable_syms = None
    if getattr(args, "reliable_only", False) and reliable_universe_path.exists():
        with open(reliable_universe_path, encoding="utf-8") as f:
            rel_data = json.load(f)
        # Desteklenen iki format: {"reliable_symbols": [...]} veya {"reliable": {...}}
        if "reliable_symbols" in rel_data:
            reliable_syms = rel_data["reliable_symbols"]
        else:
            reliable_syms = list(rel_data.get("reliable", {}).keys())
        print(f"  [RELIABLE-ONLY] {len(reliable_syms)} guvenilir hisse yuklendi")

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif getattr(args, "reliable_only", False) and reliable_syms:
        symbols = reliable_syms
    elif args.quick:
        import random
        random.seed(42)
        symbols = random.sample(DEFAULT_SYMBOLS, min(30, len(DEFAULT_SYMBOLS)))
        print(f"  [QUICK] {len(symbols)} rastgele hisse seçildi")
    elif args.sample > 0:
        import random
        random.seed(42)
        symbols = random.sample(DEFAULT_SYMBOLS, min(args.sample, len(DEFAULT_SYMBOLS)))
        print(f"  [SAMPLE={args.sample}] {len(symbols)} rastgele hisse seçildi")
    else:
        symbols = DEFAULT_SYMBOLS

    active_offsets = TEST_OFFSETS_EXTENDED if args.extended else TEST_OFFSETS
    date_range_str = "1M-12M" if args.extended else "1M-5M"

    print(f"{'='*60}")
    print(f"  BACKTEST: {len(symbols)} hisse × {len(active_offsets)} tarih noktası")
    print(f"  Tarih aralığı: {date_range_str} önce -> {HORIZON}G ileriye tahmin")
    print(f"{'='*60}\n")

    df = run_backtest(symbols, extended=args.extended)

    if df.empty:
        print("Hiç sonuç üretilemedi.")
        sys.exit(1)

    # Kaydet
    df.to_csv("c:/Users/aykut/OneDrive/Masaüstü/trade/timesfm/backtest_results.csv", index=False)

    metrics = compute_metrics(df)

    print(f"\n{'='*60}")
    print(f"  BACKTEST SONUÇLARI")
    print(f"{'='*60}")
    print(f"  Test sayısı     : {metrics['n_tests']}")
    print(f"  Hisse sayısı    : {df['symbol'].nunique()}")
    print(f"  Ort. MAPE       : {metrics['mean_mape']:.2f}%")
    print(f"  Medyan MAPE     : {metrics['median_mape']:.2f}%")
    print(f"  Yön Doğruluk    : {metrics['dir_acc']:.1f}%  ({df['dir_correct'].sum().astype(int)}/{metrics['n_tests']})")
    print(f"  Ort. MAE (USD)  : ${metrics['mae_mean']:.2f}")
    print()

    # Tarih bazında detay (market event ile)
    print("  Tarih Bazında Yön Doğruluğu:")
    date_acc = df.groupby("cutoff")["dir_correct"].agg(["mean","sum","count"]).sort_values("mean", ascending=False)
    for dt, row in date_acc.iterrows():
        event = MARKET_EVENTS.get(str(dt), "")
        bar   = "#" * int(row["mean"] * 20)
        print(f"    {dt}  {row['mean']*100:5.1f}%  [{bar:<20}]  ({int(row['sum'])}/{int(row['count'])})  {event}")

    # Sektör bazında analiz
    sector_data = []
    for sym in df["symbol"].unique():
        sec = SECTOR_MAP.get(sym, "Unknown")
        sym_df = df[df["symbol"] == sym]
        sector_data.append({"sector": sec, "dir_acc": sym_df["dir_correct"].mean(), "mape": sym_df["mape"].mean()})
    sector_df = pd.DataFrame(sector_data)
    sec_acc = sector_df.groupby("sector").agg(
        dir_acc=("dir_acc","mean"), mape=("mape","mean"), n=("dir_acc","count")
    ).sort_values("dir_acc", ascending=False)
    print()
    print("  Sektör Bazında:")
    for sec, row in sec_acc.iterrows():
        ok = "OK" if row["dir_acc"] > 0.60 else ("~" if row["dir_acc"] > 0.50 else "XX")
        print(f"    {sec[:25]:25s}  {row['dir_acc']*100:5.1f}%  MAPE={row['mape']:5.1f}%  n={int(row['n'])}  {ok}")
    print()

    print(f"  En İyi Hisse    : {metrics['best_symbol']}")
    print(f"  En Zor Hisse    : {metrics['worst_symbol']}")
    print()

    # ── Güven Eşiği Analizi ──────────────────────────────────────────────
    df["corr_pct"] = df["correction_usd"].abs() / df["price_at_cutoff"] * 100
    print("  Güven Eşiği Analizi (|XGB düzeltme| / fiyat):")
    for thresh in [0.30, 0.50, 1.00]:
        sub = df[df["corr_pct"] >= thresh]
        if len(sub) > 0:
            cov = len(sub) / len(df) * 100
            print(f"    corr_pct >= {thresh:.2f}%  →  {sub['dir_correct'].mean()*100:.1f}%  ({len(sub)} tahmin, %{cov:.0f} kapsam)")
    print()

    print("  Hisse Bazında Yön Doğruluğu (sıralı):")
    sym_dir = df.groupby("symbol")["dir_correct"].mean().sort_values(ascending=False)
    sym_mape = df.groupby("symbol")["mape"].mean()
    for sym, acc in sym_dir.items():
        mape = sym_mape[sym]
        bar = "#" * int(mape / 2)
        ok  = "OK" if acc > 0.60 else ("~" if acc > 0.50 else "XX")
        print(f"    {sym:6s} {mape:6.2f}%  {bar}  {ok}  dir={acc*100:.0f}%")

    out = "c:/Users/aykut/OneDrive/Masaüstü/trade/timesfm/backtest_dashboard.png"
    plot_dashboard(df, metrics, out)

    if args.improve:
        improve_model(df)
        print("\nModel iyileştirildi. Yeni tahminler için run_inference.py kullanın.")

    print("\nBitti. Sonraki adım:")
    print("  python backtest.py --improve   # modeli iyileştir")
    print("  python inference/run_inference.py --symbol AAPL")
