"""UEC (Uranium Energy Corp) kisa vadeli fiyat tahmini - TimesFM 2.5"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import yfinance as yf
import timesfm
from timesfm import configs
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

TICKER   = "UEC"
HORIZON  = 63          # ~3 ay ticaret gunu
CONTEXT  = 504         # ~2 yil gecmis veri

# ── 1. Veri Cekme ─────────────────────────────────────────────────────────
print(f"[1/4] {TICKER} verisi Yahoo Finance'den cekiliyor...")
raw = yf.download(TICKER, period="3y", interval="1d", progress=False, auto_adjust=True)
raw = raw["Close"].squeeze().dropna()
raw.index = pd.to_datetime(raw.index)

# Son CONTEXT kadar gunü al
prices = raw.values.astype(np.float32)[-CONTEXT:]
dates  = raw.index[-CONTEXT:]

current_price = float(prices[-1])
print(f"    Son fiyat: ${current_price:.2f}  |  Veri noktasi: {len(prices)}")

# ── 2. Model Yükleme ───────────────────────────────────────────────────────
print("[2/4] TimesFM modeli yukleniyor...")
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    device="cpu",
)
tfm.compile(configs.ForecastConfig(
    max_context=512,
    max_horizon=HORIZON,
    per_core_batch_size=8,
))
print("    Model hazir.")

# ── 3. Tahmin ──────────────────────────────────────────────────────────────
print("[3/4] Tahmin yapiliyor...")
point_fc, quant_fc = tfm.forecast(horizon=HORIZON, inputs=[prices])
point = point_fc[0]   # (63,)
quants = quant_fc[0]  # (63, 9) -> [0.1, 0.2, ..., 0.9]

# Tahmini iş günleri oluştur
last_date = dates[-1]
biz_days = pd.bdate_range(start=last_date + timedelta(days=1), periods=HORIZON)

# 1-ay (21gün) ve 3-ay (63gün) hedefleri
p1m = point[20]
p3m = point[62]
q10_3m = quants[62, 0]  # %10 alt
q90_3m = quants[62, 8]  # %90 üst
chg1m = (p1m - current_price) / current_price * 100
chg3m = (p3m - current_price) / current_price * 100

print(f"    1 Ay  tahmini: ${p1m:.2f}  ({chg1m:+.1f}%)")
print(f"    3 Ay  tahmini: ${p3m:.2f}  ({chg3m:+.1f}%)")
print(f"    3 Ay aralik : ${q10_3m:.2f} - ${q90_3m:.2f}")

# ── 4. Dashboard ───────────────────────────────────────────────────────────
print("[4/4] Grafik olusturuluyor...")

BG       = "#0d1117"
PANEL    = "#161b22"
HIST_C   = "#58a6ff"
FORE_C   = "#f0883e"
BULL_C   = "#3fb950"
BEAR_C   = "#f85149"
GRID_C   = "#21262d"
TEXT_C   = "#e6edf3"
MUTED_C  = "#8b949e"

fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.suptitle(
    f"UEC  |  Uranium Energy Corp  —  Kisa Vadeli Fiyat Tahmini  (TimesFM 2.5)",
    fontsize=14, fontweight="bold", color=TEXT_C, y=0.98
)

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[3, 1],
    hspace=0.38, wspace=0.28,
    left=0.06, right=0.97, top=0.93, bottom=0.07
)

# ─── Ana grafik (üst, tüm genişlik) ─────────────────────────────────────
ax_main = fig.add_subplot(gs[0, :])
ax_main.set_facecolor(PANEL)
ax_main.grid(True, color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.7)
for sp in ax_main.spines.values():
    sp.set_color(GRID_C)

# Son 120 günü göster (okunabilirlik)
SHOW = 120
x_hist = dates[-SHOW:]
y_hist = prices[-SHOW:]

ax_main.plot(x_hist, y_hist, color=HIST_C, linewidth=1.6,
             label="Kapaniş Fiyati", zorder=4)

# Güven bantları
ax_main.fill_between(biz_days, quants[:, 0], quants[:, 8],
                     alpha=0.12, color=FORE_C, label="%10-90 Aralik")
ax_main.fill_between(biz_days, quants[:, 1], quants[:, 7],
                     alpha=0.18, color=FORE_C, label="%20-80 Aralik")
ax_main.fill_between(biz_days, quants[:, 2], quants[:, 6],
                     alpha=0.28, color=FORE_C, label="%30-70 Aralik")

# Nokta tahmini
ax_main.plot(biz_days, point, color=FORE_C, linewidth=2,
             linestyle="--", label="Nokta Tahmini", zorder=5)

# Bağlantı
ax_main.plot([dates[-1], biz_days[0]], [prices[-1], point[0]],
             color=FORE_C, linewidth=1.5, linestyle="--", alpha=0.5)

# Dikey kesim çizgisi
ax_main.axvline(x=dates[-1], color=MUTED_C, linewidth=1.2,
                linestyle=":", alpha=0.8, label="Bugun")

# 1-ay ve 3-ay çizgileri
ax_main.axhline(y=p1m, color=BULL_C if chg1m > 0 else BEAR_C,
                linewidth=0.8, linestyle=":", alpha=0.5)
ax_main.axhline(y=p3m, color=BULL_C if chg3m > 0 else BEAR_C,
                linewidth=0.8, linestyle=":", alpha=0.5)

# Etiketler
ax_main.annotate(
    f"1A: ${p1m:.2f} ({chg1m:+.1f}%)",
    xy=(biz_days[20], p1m),
    xytext=(10, 12), textcoords="offset points",
    fontsize=9.5, color=BULL_C if chg1m > 0 else BEAR_C, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=BULL_C if chg1m > 0 else BEAR_C, lw=1),
    zorder=6
)
ax_main.annotate(
    f"3A: ${p3m:.2f} ({chg3m:+.1f}%)",
    xy=(biz_days[62], p3m),
    xytext=(-90, 18), textcoords="offset points",
    fontsize=9.5, color=BULL_C if chg3m > 0 else BEAR_C, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=BULL_C if chg3m > 0 else BEAR_C, lw=1),
    zorder=6
)

# Şu anki fiyat etiketi
ax_main.annotate(
    f"Simdi: ${current_price:.2f}",
    xy=(dates[-1], current_price),
    xytext=(-80, -22), textcoords="offset points",
    fontsize=9, color=TEXT_C, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=MUTED_C, alpha=0.9),
    zorder=7
)

ax_main.set_title(f"Fiyat Geçmişi ve 3 Aylık Tahmin  |  Son veri: {dates[-1].strftime('%d %b %Y')}",
                  color=TEXT_C, fontsize=11, pad=8)
ax_main.tick_params(colors=MUTED_C, labelsize=8)
ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax_main.set_ylabel("Fiyat (USD)", color=MUTED_C, fontsize=9)
leg = ax_main.legend(loc="upper left", fontsize=8, framealpha=0.4,
                     labelcolor=TEXT_C, facecolor=PANEL, edgecolor=GRID_C)

# ─── Alt Panel 1: Fiyat Dağılımı Histogram ───────────────────────────────
ax_hist = fig.add_subplot(gs[1, 0])
ax_hist.set_facecolor(PANEL)
ax_hist.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.5)
for sp in ax_hist.spines.values():
    sp.set_color(GRID_C)

ax_hist.hist(prices, bins=40, color=HIST_C, alpha=0.7, edgecolor="none")
ax_hist.axvline(current_price, color=FORE_C, linewidth=1.5,
                linestyle="--", label=f"Simdi ${current_price:.2f}")
ax_hist.axvline(p3m, color=BULL_C if chg3m > 0 else BEAR_C, linewidth=1.5,
                linestyle="--", label=f"3A ${p3m:.2f}")
ax_hist.set_title("Fiyat Dagilimi (2Y)", color=TEXT_C, fontsize=10)
ax_hist.tick_params(colors=MUTED_C, labelsize=7)
ax_hist.legend(fontsize=7.5, labelcolor=TEXT_C, facecolor=PANEL,
               edgecolor=GRID_C, framealpha=0.5)

# ─── Alt Panel 2: Tahmin Özet Tablosu ────────────────────────────────────
ax_tbl = fig.add_subplot(gs[1, 1])
ax_tbl.set_facecolor(PANEL)
ax_tbl.axis("off")

rows = [
    ("Simdi",   f"${current_price:.2f}",  "—",          TEXT_C),
    ("2 Hafta", f"${point[9]:.2f}",       f"{(point[9]-current_price)/current_price*100:+.1f}%",
     BULL_C if point[9] > current_price else BEAR_C),
    ("1 Ay",    f"${p1m:.2f}",            f"{chg1m:+.1f}%",
     BULL_C if chg1m > 0 else BEAR_C),
    ("2 Ay",    f"${point[41]:.2f}",      f"{(point[41]-current_price)/current_price*100:+.1f}%",
     BULL_C if point[41] > current_price else BEAR_C),
    ("3 Ay",    f"${p3m:.2f}",            f"{chg3m:+.1f}%",
     BULL_C if chg3m > 0 else BEAR_C),
    ("3A Aralik", f"${q10_3m:.2f}–${q90_3m:.2f}", "(%10-90)", MUTED_C),
]

ax_tbl.set_title("Tahmin Ozeti", color=TEXT_C, fontsize=10, pad=8)
y_pos = 0.92
for label, val, chg, color in rows:
    ax_tbl.text(0.05, y_pos, label, transform=ax_tbl.transAxes,
                fontsize=9, color=MUTED_C, va="top")
    ax_tbl.text(0.48, y_pos, val, transform=ax_tbl.transAxes,
                fontsize=9, color=TEXT_C, va="top", fontweight="bold")
    ax_tbl.text(0.82, y_pos, chg, transform=ax_tbl.transAxes,
                fontsize=9, color=color, va="top", fontweight="bold")
    y_pos -= 0.145
    ax_tbl.plot([0.03, 0.97], [y_pos + 0.07, y_pos + 0.07],
                color=GRID_C, linewidth=0.5, transform=ax_tbl.transAxes)

# ─── Alt Panel 3: Belirsizlik Bandı (3 aylık fan) ────────────────────────
ax_fan = fig.add_subplot(gs[1, 2])
ax_fan.set_facecolor(PANEL)
ax_fan.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.5)
for sp in ax_fan.spines.values():
    sp.set_color(GRID_C)

fore_x = np.arange(HORIZON)
ax_fan.fill_between(fore_x, quants[:, 0], quants[:, 8],
                    alpha=0.15, color=FORE_C, label="%10-90")
ax_fan.fill_between(fore_x, quants[:, 1], quants[:, 7],
                    alpha=0.25, color=FORE_C, label="%20-80")
ax_fan.fill_between(fore_x, quants[:, 3], quants[:, 5],
                    alpha=0.45, color=FORE_C, label="%40-60")
ax_fan.plot(fore_x, point, color=FORE_C, linewidth=2, label="Medyan")
ax_fan.axhline(current_price, color=HIST_C, linewidth=1,
               linestyle="--", alpha=0.6, label=f"Simdi ${current_price:.2f}")

ax_fan.set_title("Belirsizlik Fani (63 Gun)", color=TEXT_C, fontsize=10)
ax_fan.tick_params(colors=MUTED_C, labelsize=7)
ax_fan.set_xlabel("Ticaret Gunu", color=MUTED_C, fontsize=8)
ax_fan.legend(fontsize=7.5, labelcolor=TEXT_C, facecolor=PANEL,
              edgecolor=GRID_C, framealpha=0.5)

# Alt bilgi notu
fig.text(
    0.5, 0.005,
    "UYARI: Bu tahmin bir yapay zeka modelinin ciktisidir, yatirim tavsiyesi degildir. "
    "Gecmis performans gelecegi garantilemez.",
    ha="center", fontsize=7.5, color="#6e7681", style="italic"
)

out_path = "c:/Users/aykut/OneDrive/Masaüstü/trade/timesfm/UEC_forecast.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nKaydedildi: UEC_forecast.png")

# Konsola ozet
print("\n" + "="*45)
print(f"  UEC Tahmin Ozeti ({datetime.now().strftime('%d %b %Y')})")
print("="*45)
print(f"  Simdi      : ${current_price:.2f}")
print(f"  2 Hafta    : ${point[9]:.2f}  ({(point[9]-current_price)/current_price*100:+.1f}%)")
print(f"  1 Ay       : ${p1m:.2f}  ({chg1m:+.1f}%)")
print(f"  3 Ay       : ${p3m:.2f}  ({chg3m:+.1f}%)")
print(f"  3A %10-90  : ${q10_3m:.2f} - ${q90_3m:.2f}")
print("="*45)

plt.show()
