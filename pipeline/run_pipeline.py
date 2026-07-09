"""
UEC Tam Pipeline
================
  1. Veri  → Yahoo Finance (fiyat + haberler)
  2. Özellik → RSI, MACD, BB, Macro, Korelasyon
  3. Sentiment → VADER (haber başlıkları)
  4. TimesFM base forecast
  5. XGBoost artık düzeltme
  6. Dashboard
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

import timesfm
from timesfm import configs

from data_collector import fetch_prices, fetch_news
from features import build_features
from sentiment import build_daily_sentiment, merge_sentiment
from residual_model import ResidualCorrector, make_residual_dataset

HORIZON  = 63   # ~3 ay
CONTEXT  = 504

# ═══════════════════════════════════════════════════════════════
# 1. VERİ
# ═══════════════════════════════════════════════════════════════
print("=" * 55)
print("  UEC TAM PİPELİNE")
print("=" * 55)

print("\n[1/5] Veri cekiliyor...")
prices_df = fetch_prices(period="3y")
news_df   = fetch_news("UEC")
print(f"  Fiyat: {prices_df.shape}  |  Haber: {len(news_df)}")

# ═══════════════════════════════════════════════════════════════
# 2. ÖZELLİKLER
# ═══════════════════════════════════════════════════════════════
print("\n[2/5] Ozellikler uretiliyor...")
features = build_features(prices_df)

# Sentiment ekle
if not news_df.empty:
    sentiment = build_daily_sentiment(news_df, use_finbert=False)
    features  = merge_sentiment(features, sentiment)
    print(f"  Sentiment: {features['Sentiment'].notna().sum()} gun skoru")
else:
    features["Sentiment"] = 0.0
    print("  Haber bulunamadi, Sentiment=0")

print(f"  Toplam ozellik: {features.shape[1]}  |  Satir: {features.shape[0]}")
print(f"  Kolonlar: {', '.join(features.columns.tolist())}")

# ═══════════════════════════════════════════════════════════════
# 3. TIMESFM BASE FORECAST
# ═══════════════════════════════════════════════════════════════
print("\n[3/5] TimesFM modeli yukleniyor...")
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    device="cpu",
)
tfm.compile(configs.ForecastConfig(
    max_context=512,
    max_horizon=HORIZON,
    per_core_batch_size=8,
))
print("  Model hazir.")

uec_prices = prices_df["UEC"].dropna()
prices_arr = uec_prices.values.astype(np.float32)[-CONTEXT:]
dates_arr  = uec_prices.index[-CONTEXT:]
current    = float(prices_arr[-1])

def tfm_forecast_fn(ctx: np.ndarray) -> np.ndarray:
    pts, _ = tfm.forecast(horizon=HORIZON, inputs=[ctx.astype(np.float32)])
    return pts[0]

base_point, base_quants = tfm.forecast(horizon=HORIZON, inputs=[prices_arr])
base_point  = base_point[0]
base_quants = base_quants[0]

from datetime import timedelta
last_date = uec_prices.index[-1]
biz_days  = pd.bdate_range(start=last_date + timedelta(days=1), periods=HORIZON)

print(f"  Base 1A: ${base_point[20]:.2f}  |  Base 3A: ${base_point[62]:.2f}")

# ═══════════════════════════════════════════════════════════════
# 4. XGBOOST ARTIK DÜZELTMESİ
# ═══════════════════════════════════════════════════════════════
print("\n[4/5] XGBoost artik modeli egitiliyor...")
print("  (Walk-forward pencereler olusturuluyor, ~1-2 dk)")

X, y, train_dates = make_residual_dataset(
    features_df     = features,
    uec_prices      = uec_prices,
    tfm_forecast_fn = tfm_forecast_fn,
    horizon         = 21,    # 1 ay artık öğren
    context         = 252,
    step            = 5,
)

corrector = ResidualCorrector()

if len(X) >= 20:
    cv_result = corrector.fit(X, y)
    print(f"  Egitim ornegi: {len(X)}")
    print(f"  CV MAE: {cv_result['cv_mae_mean']:.3f} ± {cv_result['cv_mae_std']:.3f}")

    # Mevcut an için artık tahmini (eğitimde kullanılan kolonlar)
    feat_cols = list(features.columns)
    latest_feat = features[feat_cols].iloc[-1].values
    correction  = corrector.predict(latest_feat)

    # Özellik önemi
    imp = corrector.feature_importance(feat_cols)
    top5 = imp.head(5)
    print(f"\n  En etkili 5 faktor:")
    for f, v in top5.items():
        print(f"    {f:20s} {v:.3f}")
else:
    correction = 0.0
    print(f"  Yetersiz egitim verisi ({len(X)}), duzeltme=0")

# Düzeltilmiş tahmin
corrected_point = base_point + correction
corrected_point = np.maximum(corrected_point, 0.5)  # negatif fiyat olmasın

print(f"\n  Base 3A      : ${base_point[62]:.2f}")
print(f"  Duzeltme     : {correction:+.2f}")
print(f"  Duzeltilmis  : ${corrected_point[62]:.2f}")

# ═══════════════════════════════════════════════════════════════
# 5. DASHBOARD
# ═══════════════════════════════════════════════════════════════
print("\n[5/5] Dashboard olusturuluyor...")

BG, PANEL = "#0d1117", "#161b22"
HIST_C    = "#58a6ff"
BASE_C    = "#6e7681"
CORR_C    = "#f0883e"
BULL_C    = "#3fb950"
BEAR_C    = "#f85149"
GRID_C    = "#21262d"
TEXT_C    = "#e6edf3"
MUTED_C   = "#8b949e"

fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle(
    "UEC — Tam Pipeline Tahmini  (TimesFM 2.5 + Teknik + Sentiment + XGBoost)",
    fontsize=14, fontweight="bold", color=TEXT_C, y=0.985
)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       height_ratios=[3, 1.2, 1.2],
                       hspace=0.48, wspace=0.30,
                       left=0.06, right=0.97, top=0.96, bottom=0.05)

# ── Ana Grafik ────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
ax.set_facecolor(PANEL)
ax.grid(True, color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)
for sp in ax.spines.values(): sp.set_color(GRID_C)

SHOW = 150
ax.plot(dates_arr[-SHOW:], prices_arr[-SHOW:],
        color=HIST_C, linewidth=1.6, label="Gecmis Fiyat", zorder=4)

# Güven bantları (base quantiles)
ax.fill_between(biz_days, base_quants[:, 0], base_quants[:, 8],
                alpha=0.10, color=BASE_C, label="Base %10-90")
ax.fill_between(biz_days, base_quants[:, 2], base_quants[:, 6],
                alpha=0.18, color=BASE_C)

# Base forecast
ax.plot(biz_days, base_point, color=BASE_C, linewidth=1.5,
        linestyle="--", alpha=0.7, label="TimesFM base", zorder=4)

# Düzeltilmiş forecast
ax.plot(biz_days, corrected_point, color=CORR_C, linewidth=2.2,
        linestyle="-", label="Duzeltilmis Tahmin", zorder=5)

# Bağlantı
ax.plot([dates_arr[-1], biz_days[0]], [current, corrected_point[0]],
        color=CORR_C, linewidth=1.5, linestyle="-", alpha=0.5)

ax.axvline(x=dates_arr[-1], color=MUTED_C, linewidth=1,
           linestyle=":", alpha=0.7, label="Bugun")

# Etiketler
for label, idx, xoff, yoff in [
    ("2H", 9,  15, 12),
    ("1A", 20, 10, 14),
    ("3A", 62, -80, 16),
]:
    val = corrected_point[idx]
    chg = (val - current) / current * 100
    clr = BULL_C if chg > 0 else BEAR_C
    ax.annotate(
        f"{label}: ${val:.2f} ({chg:+.1f}%)",
        xy=(biz_days[idx], val),
        xytext=(xoff, yoff), textcoords="offset points",
        fontsize=9, color=clr, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=clr, lw=0.9), zorder=6
    )

ax.annotate(f"Simdi: ${current:.2f}",
            xy=(dates_arr[-1], current),
            xytext=(-75, -20), textcoords="offset points",
            fontsize=9, color=TEXT_C, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                      edgecolor=MUTED_C, alpha=0.9), zorder=7)

ax.set_title(f"UEC Fiyat Tahmini  |  Son veri: {dates_arr[-1].strftime('%d %b %Y')}  "
             f"|  Duzeltme: {correction:+.2f} USD",
             color=TEXT_C, fontsize=11, pad=8)
ax.tick_params(colors=MUTED_C, labelsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.set_ylabel("Fiyat (USD)", color=MUTED_C, fontsize=9)
ax.legend(loc="upper left", fontsize=8, framealpha=0.4,
          labelcolor=TEXT_C, facecolor=PANEL, edgecolor=GRID_C)

# ── RSI Paneli ────────────────────────────────────────────────
ax_rsi = fig.add_subplot(gs[1, 0])
ax_rsi.set_facecolor(PANEL)
ax_rsi.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.5)
for sp in ax_rsi.spines.values(): sp.set_color(GRID_C)

rsi_vals = features["RSI"].iloc[-SHOW:]
clrs = [BULL_C if v < 30 else BEAR_C if v > 70 else HIST_C for v in rsi_vals]
ax_rsi.bar(rsi_vals.index, rsi_vals, color=clrs, width=1, alpha=0.8)
ax_rsi.axhline(70, color=BEAR_C, linewidth=0.8, linestyle="--", alpha=0.6)
ax_rsi.axhline(30, color=BULL_C, linewidth=0.8, linestyle="--", alpha=0.6)
ax_rsi.set_ylim(0, 100)
ax_rsi.set_title(f"RSI-14  (Son: {rsi_vals.iloc[-1]:.1f})", color=TEXT_C, fontsize=10)
ax_rsi.tick_params(colors=MUTED_C, labelsize=7)
ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# ── MACD Paneli ────────────────────────────────────────────────
ax_macd = fig.add_subplot(gs[1, 1])
ax_macd.set_facecolor(PANEL)
ax_macd.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.5)
for sp in ax_macd.spines.values(): sp.set_color(GRID_C)

macd_hist = features["MACD_hist"].iloc[-SHOW:]
bar_clrs = [BULL_C if v >= 0 else BEAR_C for v in macd_hist]
ax_macd.bar(macd_hist.index, macd_hist, color=bar_clrs, width=1, alpha=0.8)
ax_macd.plot(features["MACD"].iloc[-SHOW:].index,
             features["MACD"].iloc[-SHOW:], color="#a371f7", linewidth=1)
ax_macd.plot(features["MACD_sig"].iloc[-SHOW:].index,
             features["MACD_sig"].iloc[-SHOW:], color="#f0883e", linewidth=1)
ax_macd.axhline(0, color=MUTED_C, linewidth=0.5)
ax_macd.set_title("MACD (12,26,9)", color=TEXT_C, fontsize=10)
ax_macd.tick_params(colors=MUTED_C, labelsize=7)
ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# ── Sentiment Paneli ───────────────────────────────────────────
ax_sent = fig.add_subplot(gs[1, 2])
ax_sent.set_facecolor(PANEL)
ax_sent.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.5)
for sp in ax_sent.spines.values(): sp.set_color(GRID_C)

sent_vals = features["Sentiment"].iloc[-SHOW:]
s_clrs = [BULL_C if v > 0 else BEAR_C if v < 0 else MUTED_C for v in sent_vals]
ax_sent.bar(sent_vals.index, sent_vals, color=s_clrs, width=1, alpha=0.8)
ax_sent.axhline(0, color=MUTED_C, linewidth=0.5)
ax_sent.set_title(f"Haber Sentiment (VADER)  Son: {sent_vals.iloc[-1]:.3f}",
                  color=TEXT_C, fontsize=10)
ax_sent.tick_params(colors=MUTED_C, labelsize=7)
ax_sent.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

# ── Özet Tablo ────────────────────────────────────────────────
ax_tbl = fig.add_subplot(gs[2, 0])
ax_tbl.set_facecolor(PANEL)
ax_tbl.axis("off")
ax_tbl.set_title("Tahmin Ozeti", color=TEXT_C, fontsize=10, pad=6)

rows = [
    ("Simdi",    f"${current:.2f}",          "—",       TEXT_C),
    ("2 Hafta",  f"${corrected_point[9]:.2f}",
     f"{(corrected_point[9]-current)/current*100:+.1f}%",
     BULL_C if corrected_point[9] > current else BEAR_C),
    ("1 Ay",     f"${corrected_point[20]:.2f}",
     f"{(corrected_point[20]-current)/current*100:+.1f}%",
     BULL_C if corrected_point[20] > current else BEAR_C),
    ("2 Ay",     f"${corrected_point[41]:.2f}",
     f"{(corrected_point[41]-current)/current*100:+.1f}%",
     BULL_C if corrected_point[41] > current else BEAR_C),
    ("3 Ay",     f"${corrected_point[62]:.2f}",
     f"{(corrected_point[62]-current)/current*100:+.1f}%",
     BULL_C if corrected_point[62] > current else BEAR_C),
    ("XGB Duz.", f"{correction:+.3f}",        "artik",   MUTED_C),
]
y0 = 0.93
for label, val, chg, clr in rows:
    ax_tbl.text(0.04, y0, label, transform=ax_tbl.transAxes,
                fontsize=9, color=MUTED_C, va="top")
    ax_tbl.text(0.48, y0, val,  transform=ax_tbl.transAxes,
                fontsize=9, color=TEXT_C, va="top", fontweight="bold")
    ax_tbl.text(0.82, y0, chg,  transform=ax_tbl.transAxes,
                fontsize=9, color=clr, va="top", fontweight="bold")
    y0 -= 0.14

# ── Özellik Önemi ─────────────────────────────────────────────
ax_imp = fig.add_subplot(gs[2, 1])
ax_imp.set_facecolor(PANEL)
ax_imp.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.5, axis="x")
for sp in ax_imp.spines.values(): sp.set_color(GRID_C)
ax_imp.set_title("XGBoost Faktor Onemi", color=TEXT_C, fontsize=10)

if len(X) >= 20:
    feat_cols_plot = list(features.columns)
    imp = corrector.feature_importance(feat_cols_plot).head(8)
    bars = ax_imp.barh(imp.index[::-1], imp.values[::-1],
                       color=CORR_C, alpha=0.8, edgecolor="none")
    ax_imp.tick_params(colors=MUTED_C, labelsize=7.5)
else:
    ax_imp.text(0.5, 0.5, "Yetersiz veri", ha="center", va="center",
                color=MUTED_C, transform=ax_imp.transAxes)

# ── Makro Korelasyon ──────────────────────────────────────────
ax_corr = fig.add_subplot(gs[2, 2])
ax_corr.set_facecolor(PANEL)
ax_corr.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.5, axis="x")
for sp in ax_corr.spines.values(): sp.set_color(GRID_C)
ax_corr.set_title("Macro Korelasyon (60G)", color=TEXT_C, fontsize=10)

macro_cols = ["URA", "CCJ", "DXY", "VIX", "TLT"]
corr_vals = {
    col: float(prices_df["UEC"].tail(60).corr(prices_df[col].tail(60)))
    for col in macro_cols if col in prices_df.columns
}
corr_s = pd.Series(corr_vals).sort_values()
clrs_c = [BULL_C if v > 0 else BEAR_C for v in corr_s]
ax_corr.barh(corr_s.index, corr_s.values, color=clrs_c, alpha=0.8, edgecolor="none")
ax_corr.axvline(0, color=MUTED_C, linewidth=0.7)
ax_corr.set_xlim(-1, 1)
ax_corr.tick_params(colors=MUTED_C, labelsize=8)

fig.text(0.5, 0.01,
         "UYARI: Yapay zeka tahminidir, yatirim tavsiyesi degildir.",
         ha="center", fontsize=7.5, color="#6e7681", style="italic")

out = "c:/Users/aykut/OneDrive/Masaüstü/trade/timesfm/UEC_full_pipeline.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nKaydedildi: UEC_full_pipeline.png")

print("\n" + "=" * 45)
print(f"  UEC TAM PİPELİNE SONUÇLARI")
print("=" * 45)
print(f"  Simdi    : ${current:.2f}")
p2h = corrected_point[9]
p1a = corrected_point[20]
p3a = corrected_point[62]
print(f"  2 Hafta  : ${p2h:.2f} ({(p2h-current)/current*100:+.1f}%)")
print(f"  1 Ay     : ${p1a:.2f} ({(p1a-current)/current*100:+.1f}%)")
print(f"  3 Ay     : ${p3a:.2f} ({(p3a-current)/current*100:+.1f}%)")
print(f"  XGB duz. : {correction:+.3f} USD")
print("=" * 45)

plt.show()
