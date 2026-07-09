"""TimesFM Görsel Dashboard - 3 farklı senaryo tahmini."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import timesfm
from timesfm import configs

# ── Model yükleme ──────────────────────────────────────────────────────────
print("Model yukleniyor...")
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    device="cpu",
)
tfm.compile(configs.ForecastConfig(
    max_context=256,
    max_horizon=48,
    per_core_batch_size=8,
))
print("Model hazir.\n")

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
np.random.seed(7)

# ── 3 farklı senaryo verisi ────────────────────────────────────────────────
scenarios = {}

# 1. Mevsimsel Satis (haftalik periyot)
t = np.arange(150)
trend = 100 + 0.3 * t
seasonal = 20 * np.sin(2 * np.pi * t / 52)
noise = np.random.randn(150) * 5
scenarios["Magaza Satislari (Haftalik)"] = trend + seasonal + noise

# 2. Gunluk Sicaklik (yillik periyot)
t = np.arange(180)
temp = 15 + 12 * np.sin(2 * np.pi * t / 365 * 12 - np.pi / 2)
noise = np.random.randn(180) * 1.5
scenarios["Gunluk Sicaklik (C)"] = temp + noise

# 3. Hisse Senedi Benzeri Seri (random walk + trend)
t = np.arange(120)
rw = np.cumsum(np.random.randn(120) * 2)
price = 50 + rw + 0.05 * t
scenarios["Hisse Fiyati (TL)"] = price

# ── Tahmin ────────────────────────────────────────────────────────────────
HORIZON = 48
results = {}
for name, series in scenarios.items():
    pts, quants = tfm.forecast(horizon=HORIZON, inputs=[series])
    results[name] = {
        "series": series,
        "point": pts[0],
        "quants": quants[0],  # shape: (HORIZON, 9)
    }
    print(f"Tahmin tamam: {name}")

# ── Dashboard çizimi ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor="#0f1117")
fig.suptitle(
    "TimesFM 2.5 — Zaman Serisi Tahmin Dashboardu",
    fontsize=16, fontweight="bold", color="white", y=0.98
)

gs = gridspec.GridSpec(3, 1, hspace=0.55, left=0.07, right=0.97, top=0.92, bottom=0.06)

COLORS = {
    "hist":    "#4fc3f7",
    "point":   "#ff7043",
    "q10_90":  "#ff7043",
    "q20_80":  "#ff7043",
    "q30_70":  "#ff7043",
    "vline":   "#888888",
    "grid":    "#2a2a3a",
    "bg":      "#1a1a2e",
}

for idx, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[idx])
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#333355")

    series = res["series"]
    point  = res["point"]
    quants = res["quants"]  # (48, 9)
    n = len(series)

    x_hist = np.arange(n)
    x_fore = np.arange(n, n + HORIZON)

    # Geçmiş veri
    ax.plot(x_hist, series, color=COLORS["hist"], linewidth=1.5,
            label="Gecmis Veri", zorder=3)

    # Güven bantları (%10-90, %20-80, %30-70)
    ax.fill_between(x_fore, quants[:, 0], quants[:, 8],  # 10-90
                    alpha=0.15, color=COLORS["q10_90"], label="%10-90 aralik")
    ax.fill_between(x_fore, quants[:, 1], quants[:, 7],  # 20-80
                    alpha=0.20, color=COLORS["q20_80"], label="%20-80 aralik")
    ax.fill_between(x_fore, quants[:, 2], quants[:, 6],  # 30-70
                    alpha=0.30, color=COLORS["q30_70"], label="%30-70 aralik")

    # Nokta tahmini
    ax.plot(x_fore, point, color=COLORS["point"], linewidth=2,
            linestyle="--", label="Nokta Tahmini", zorder=4)

    # Kesim çizgisi
    ax.axvline(x=n - 1, color=COLORS["vline"], linewidth=1.2,
               linestyle=":", alpha=0.8, label="Tahmin Baslangici")

    # Son gerçek noktayı bağla
    ax.plot([n - 1, n], [series[-1], point[0]],
            color=COLORS["point"], linewidth=1.5, linestyle="--", alpha=0.6)

    ax.set_title(name, color="white", fontsize=12, pad=8, fontweight="bold")
    ax.tick_params(colors="gray", labelsize=8)
    ax.set_xlabel("Adim", color="gray", fontsize=8)

    # Legend
    leg = ax.legend(loc="upper left", fontsize=7.5, framealpha=0.3,
                    labelcolor="white", facecolor="#111122", edgecolor="#333355")

    # Son 3 tahmin değerini etiketle
    for step in [11, 23, 47]:
        if step < HORIZON:
            ax.annotate(
                f"t+{step+1}: {point[step]:.1f}",
                xy=(x_fore[step], point[step]),
                xytext=(5, 10), textcoords="offset points",
                fontsize=7.5, color="#ffcc80",
                arrowprops=dict(arrowstyle="->", color="#ffcc80", lw=0.8),
            )

plt.savefig(
    "c:/Users/aykut/OneDrive/Masaüstü/trade/timesfm/forecast_dashboard.png",
    dpi=150, bbox_inches="tight", facecolor="#0f1117"
)
print("\nDashboard kaydedildi: forecast_dashboard.png")
plt.show()
