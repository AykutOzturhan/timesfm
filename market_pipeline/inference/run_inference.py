"""
Herhangi bir ABD hissesi için tahmin — eğitilmiş pipeline kullanır.

Kullanım:
  python inference/run_inference.py --symbol AAPL
  python inference/run_inference.py --symbol UEC --horizon 21
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import urllib.request, json as _json
import warnings
warnings.filterwarnings("ignore")

from predictor import MarketPredictor


def _day_after_risk(symbol: str) -> dict:
    """Dünkü tam gün hareketi + bugünkü intraday momentum.
    Piyasa saatlerinde son yfinance barı in-progress olduğundan
    [-2]/[-3] = dünkü kapanış, [-1]/[-2] = bugünkü intraday.
    """
    try:
        import yfinance as yf
        df = yf.download(symbol, period="7d", interval="1d",
                         auto_adjust=True, progress=False)
        c = df["Close"].squeeze().dropna()
        if len(c) < 3:
            return {}
        yesterday_chg = (float(c.iloc[-2]) / float(c.iloc[-3]) - 1) * 100
        today_chg     = (float(c.iloc[-1]) / float(c.iloc[-2]) - 1) * 100
        return {"prev_change_pct": yesterday_chg, "today_change_pct": today_chg}
    except Exception:
        return {}


def _polygon_options_signal(symbol: str) -> dict | None:
    """Polygon.io'dan bugünkü options sinyali — P/C ratio, IV, sentiment."""
    try:
        from config import POLYGON_API_KEY
        url = (f"https://api.polygon.io/v3/snapshot/options/{symbol}"
               f"?limit=250&apiKey={POLYGON_API_KEY}")
        with urllib.request.urlopen(url, timeout=8) as r:
            data = _json.loads(r.read())
        results = data.get("results", [])
        if not results:
            return None
        calls = [x for x in results if x["details"]["contract_type"] == "call"]
        puts  = [x for x in results if x["details"]["contract_type"] == "put"]
        call_vol = sum(x.get("day", {}).get("volume", 0) for x in calls)
        put_vol  = sum(x.get("day", {}).get("volume", 0) for x in puts)
        call_oi  = sum(x.get("open_interest", 0) for x in calls)
        put_oi   = sum(x.get("open_interest", 0) for x in puts)
        ivs = [x.get("implied_volatility", 0) for x in results if x.get("implied_volatility")]
        avg_iv = sum(ivs) / len(ivs) if ivs else 0
        vol_pc = put_vol / max(call_vol, 1)
        oi_pc  = put_oi  / max(call_oi, 1)
        if vol_pc < 0.60:
            sentiment, slabel = "BULLISH", "Opsiyonlar YUKSELIS bekliyor"
        elif vol_pc > 1.30:
            sentiment, slabel = "BEARISH", "Opsiyonlar DUSUS bekliyor"
        else:
            sentiment, slabel = "NOTR", "Opsiyonlar NOTR"
        return {
            "vol_pc": vol_pc, "oi_pc": oi_pc, "avg_iv": avg_iv,
            "sentiment": sentiment, "label": slabel,
            "call_vol": call_vol, "put_vol": put_vol,
        }
    except Exception:
        return None

parser = argparse.ArgumentParser()
parser.add_argument("--symbol",  default="AAPL", help="Hisse kodu")
parser.add_argument("--sector",  default="",     help="GICS sektör (opsiyonel)")
parser.add_argument("--horizon", type=int, default=63, help="Kaç iş günü (default 63=3ay)")
parser.add_argument("--min-conf", type=float, default=0.0,
                    help="Min güven eşiği (corr_pct). 0.5=yüksek güven, 1.0=çok yüksek güven")
args = parser.parse_args()

SYM     = args.symbol.upper()
HORIZON = args.horizon

print(f"\nTahmin: {SYM}  |  Horizon: {HORIZON} is gunu")

predictor = MarketPredictor()
result = predictor.predict(SYM, sector=args.sector, horizon=HORIZON)

current   = result["current"]
corrected = np.array(result["corrected"])
base      = np.array(result["base_forecast"])
quants    = np.array(result["quantiles"])
biz_days  = result["biz_days"]
milestones = result["milestones"]
correction = result["correction_usd"]

# ── Güven Skoru ───────────────────────────────────────────────────────────
corr_pct = abs(correction) / current * 100   # % cinsinden XGB düzeltme büyüklüğü

# Güven seviyesi sınıflandırması (backtest analizi: >=0.50% → 78.6%, >=1.00% → 90.8%)
if corr_pct >= 1.00:
    conf_label = "COK YUKSEK (>=1.00%)"
    conf_acc   = "~90%"
elif corr_pct >= 0.50:
    conf_label = "YUKSEK (>=0.50%)"
    conf_acc   = "~79%"
elif corr_pct >= 0.30:
    conf_label = "ORTA (>=0.30%)"
    conf_acc   = "~76%"
else:
    conf_label = "DUSUK (<0.30%)"
    conf_acc   = "~65%"

# Reliable universe kontrolü
import json
_rel_path = Path(__file__).parent.parent / "data" / "reliable_universe.json"
in_reliable = False
if _rel_path.exists():
    _rel = json.loads(_rel_path.read_text())
    in_reliable = SYM in _rel.get("reliable_symbols", [])

# ── Konsol Özet ───────────────────────────────────────────────────────────
print(f"\n{'='*52}")
print(f"  {SYM} TAHMIN SONUCLARI")
print(f"{'='*52}")
print(f"  Simdi          : ${current:.2f}")
for label, m in milestones.items():
    arrow = "^" if m["change_pct"] > 0 else "v"
    print(f"  {label:12s}   : ${m['price']:.2f}  {arrow} {m['change_pct']:+.1f}%")
print(f"  XGB duzeltme   : {correction:+.3f} USD  (corr_pct={corr_pct:.2f}%)")
print(f"  Guven Seviyesi : {conf_label}  ->  tarihsel dogruluk {conf_acc}")
print(f"  Guvenilir Evren: {'EVET' if in_reliable else 'HAYIR'}")

# Polygon.io anlık options sinyali
_opts = _polygon_options_signal(SYM)
if _opts:
    _pc  = _opts["vol_pc"]
    _iv  = _opts["avg_iv"]
    _lbl = _opts["label"]
    print(f"  Options (bugun)  : P/C={_pc:.2f}  IV={_iv:.0%}  -- {_lbl}")
    # Model + Options kombinasyonu
    _model_dir = "YUKSELIS" if correction > 0 else "DUSUS"
    _opts_dir  = _opts["sentiment"]
    if (_model_dir == "YUKSELIS" and _opts_dir == "BULLISH") or \
       (_model_dir == "DUSUS"   and _opts_dir == "BEARISH"):
        print(f"  ** Model + Options UYUMLU -> Sinyal Gucu: COK YUKSEK **")
    elif _opts_dir == "NOTR":
        print(f"  Model yonu: {_model_dir}  |  Options notr")
    else:
        print(f"  UYARI: Model {_model_dir} ama Options {_opts_dir} -> Karisik sinyal")

# ── 0DTE Risk Kontrol ─────────────────────────────────────────────────────
_risk      = _day_after_risk(SYM)
_prev_chg  = _risk.get("prev_change_pct", 0.0)
_today_chg = _risk.get("today_change_pct", 0.0)
_warn_lines = []

# Kural 1: Büyük rally sonrası gün (dünkü kapanış +2.5%+)
if _prev_chg > 2.5:
    _warn_lines.append(
        f"  [!] ONCEKI GUN RALLY: {_prev_chg:+.1f}%  — Rally sonrasi gun;"
        f" konsolidasyon/gerikilis riski YUKSEK. Cok dikkatli ol."
    )

if _opts:
    _pc = _opts["vol_pc"]
    # Kural 2: Aşırı düşük P/C + önceki gün büyük rally = FOMO tuzağı
    if _pc < 0.25 and _prev_chg > 2.0:
        _warn_lines.append(
            f"  [!!] FOMO TUZAGI: P/C={_pc:.2f} + onceki gun {_prev_chg:+.1f}%"
            f" — Piyasa yapicilar CALL SATIYOR olabilir. 0DTE CALL ALMA."
        )
    elif _pc < 0.25:
        _warn_lines.append(
            f"  [!] P/C={_pc:.2f} cok dusuk — Kurumsal al mi, retail FOMO mu?"
            f" ORB kirilimsiz girme."
        )
    # Kural 3: P/C bullish ama fiyat hareketi zayıf (flat piyasada OTM call tehlikeli)
    if _pc < 0.35 and abs(_today_chg) < 0.5:
        _warn_lines.append(
            f"  [!] P/C BULLISH ama fiyat FLAT ({_today_chg:+.1f}%) — OTM 0DTE call"
            f" tehlikeli! Sadece ORB kirilimindan sonra gir."
        )
    # Kural 4: Genel 0DTE ORB zorunluluğu
    if _prev_chg > 1.5 or _pc < 0.30:
        _warn_lines.append(
            f"  0DTE KURAL: Ilk 15dk ORB araligindan once GIRME."
            f" Kirilim + hacim onayini bekle."
        )

if _warn_lines:
    print(f"  {'-'*48}")
    print(f"  0DTE RISK KONTROL:")
    for w in _warn_lines:
        print(w)

print(f"{'='*52}")

# Güven eşiği filtresi
if args.min_conf > 0 and corr_pct < args.min_conf:
    print(f"\n  UYARI: corr_pct={corr_pct:.2f}% < min_conf={args.min_conf:.2f}%")
    print(f"  Bu hisse icin model yeterince guvenli degil, tahmin atlanıyor.")
    sys.exit(0)

# ── Dashboard ─────────────────────────────────────────────────────────────
BG, PANEL = "#0d1117", "#161b22"
HIST_C    = "#58a6ff"
BASE_C    = "#6e7681"
CORR_C    = "#f0883e"
BULL_C    = "#3fb950"
BEAR_C    = "#f85149"
GRID_C    = "#21262d"
TEXT_C    = "#e6edf3"
MUTED_C   = "#8b949e"

fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.suptitle(f"{SYM}  —  Piyasa Pipeline Tahmini (TimesFM 2.5 + XGBoost)",
             fontsize=14, fontweight="bold", color=TEXT_C, y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       height_ratios=[3, 1.2],
                       hspace=0.40, wspace=0.28,
                       left=0.06, right=0.97, top=0.93, bottom=0.06)

# Ana grafik
ax = fig.add_subplot(gs[0, :])
ax.set_facecolor(PANEL)
ax.grid(True, color=GRID_C, linewidth=0.5, linestyle="--", alpha=0.6)
for sp in ax.spines.values(): sp.set_color(GRID_C)

# Geçmiş fiyat (son 120 gün)
feats = result["features"]
hist_close = feats.get("UEC", pd.Series()) if "UEC" in (feats.columns if hasattr(feats, "columns") else []) else None

# yfinance'den doğrudan al
import yfinance as yf

raw = yf.download(SYM, period="6mo", interval="1d", auto_adjust=True, progress=False)
hist = raw["Close"].squeeze().dropna()
ax.plot(hist.index, hist.values, color=HIST_C, linewidth=1.6,
        label="Kapaniş Fiyati", zorder=4)

# Güven bantları
ax.fill_between(biz_days, quants[:, 0], quants[:, 8],
                alpha=0.10, color=BASE_C, label="Base %10-90")
ax.fill_between(biz_days, quants[:, 2], quants[:, 6],
                alpha=0.18, color=BASE_C)
ax.plot(biz_days, base, color=BASE_C, linewidth=1.3,
        linestyle="--", alpha=0.6, label="TimesFM base")
ax.plot(biz_days, corrected, color=CORR_C, linewidth=2.2,
        label="Duzeltilmis Tahmin", zorder=5)
ax.plot([hist.index[-1], biz_days[0]], [hist.values[-1], corrected[0]],
        color=CORR_C, linewidth=1.5, alpha=0.5)
ax.axvline(x=hist.index[-1], color=MUTED_C, linewidth=1, linestyle=":", alpha=0.7)

for label, m in milestones.items():
    idx_map = {"2W": 9, "1M": 20, "2M": 41, "3M": 62}
    idx = idx_map.get(label, 20)
    if idx < len(corrected):
        clr = BULL_C if m["change_pct"] > 0 else BEAR_C
        ax.annotate(
            f"{label}: ${m['price']:.2f} ({m['change_pct']:+.1f}%)",
            xy=(biz_days[idx], corrected[idx]),
            xytext=(8, 14), textcoords="offset points",
            fontsize=9, color=clr, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=clr, lw=0.9)
        )

ax.annotate(f"Simdi: ${current:.2f}",
            xy=(hist.index[-1], current),
            xytext=(-80, -20), textcoords="offset points",
            fontsize=9, color=TEXT_C, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                      edgecolor=MUTED_C, alpha=0.9))

ax.set_title(f"{SYM} | Horizon: {HORIZON} iş günü | XGB düzeltme: {correction:+.3f} USD",
             color=TEXT_C, fontsize=11, pad=8)
ax.tick_params(colors=MUTED_C, labelsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.set_ylabel("Fiyat (USD)", color=MUTED_C, fontsize=9)
ax.legend(loc="upper left", fontsize=8, framealpha=0.4,
          labelcolor=TEXT_C, facecolor=PANEL, edgecolor=GRID_C)

# Alt: Tahmin özeti tablosu
ax_tbl = fig.add_subplot(gs[1, 0])
ax_tbl.set_facecolor(PANEL)
ax_tbl.axis("off")
ax_tbl.set_title("Tahmin Ozeti", color=TEXT_C, fontsize=10, pad=6)
rows = [("Simdi", f"${current:.2f}", "—", TEXT_C)]
for label, m in milestones.items():
    clr = BULL_C if m["change_pct"] > 0 else BEAR_C
    rows.append((label, f"${m['price']:.2f}",
                 f"{m['change_pct']:+.1f}%", clr))
rows.append(("XGB Duz.", f"{correction:+.3f}", "USD", MUTED_C))
y0 = 0.92
for lbl, val, chg, clr in rows:
    ax_tbl.text(0.04, y0, lbl, transform=ax_tbl.transAxes,
                fontsize=9, color=MUTED_C, va="top")
    ax_tbl.text(0.48, y0, val, transform=ax_tbl.transAxes,
                fontsize=9, color=TEXT_C, va="top", fontweight="bold")
    ax_tbl.text(0.82, y0, chg, transform=ax_tbl.transAxes,
                fontsize=9, color=clr, va="top", fontweight="bold")
    y0 -= 0.16

# Alt: Belirsizlik fanı
ax_fan = fig.add_subplot(gs[1, 1])
ax_fan.set_facecolor(PANEL)
ax_fan.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.5)
for sp in ax_fan.spines.values(): sp.set_color(GRID_C)
x = np.arange(HORIZON)
ax_fan.fill_between(x, quants[:, 0], quants[:, 8], alpha=0.15, color=CORR_C)
ax_fan.fill_between(x, quants[:, 2], quants[:, 6], alpha=0.30, color=CORR_C)
ax_fan.plot(x, corrected, color=CORR_C, linewidth=2)
ax_fan.axhline(current, color=HIST_C, linewidth=1, linestyle="--", alpha=0.6)
ax_fan.set_title("Belirsizlik Fani", color=TEXT_C, fontsize=10)
ax_fan.tick_params(colors=MUTED_C, labelsize=7)
ax_fan.set_xlabel("Is Gunu", color=MUTED_C, fontsize=8)

# Alt: Getiri dağılımı tahmini
ax_dist = fig.add_subplot(gs[1, 2])
ax_dist.set_facecolor(PANEL)
ax_dist.grid(True, color=GRID_C, linewidth=0.4, linestyle="--", alpha=0.5)
for sp in ax_dist.spines.values(): sp.set_color(GRID_C)
final_returns = (quants[-1, :] - current) / current * 100
n_q = len(final_returns)
qs  = np.linspace(10, 90, n_q)
clrs_q = [BULL_C if r > 0 else BEAR_C for r in final_returns]
ax_dist.bar(qs, final_returns, color=clrs_q, alpha=0.8, width=max(5, 80//n_q))
ax_dist.axhline(0, color=MUTED_C, linewidth=0.7)
ax_dist.set_title(f"3A Getiri Dagilimi (%)", color=TEXT_C, fontsize=10)
ax_dist.set_xlabel("Quantile", color=MUTED_C, fontsize=8)
ax_dist.tick_params(colors=MUTED_C, labelsize=7)

fig.text(0.5, 0.01,
         "UYARI: Yapay zeka tahminidir, yatirim tavsiyesi degildir.",
         ha="center", fontsize=7.5, color="#6e7681", style="italic")

out = f"c:/Users/aykut/OneDrive/Masaüstü/trade/timesfm/{SYM}_market_pipeline.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nKaydedildi: {SYM}_market_pipeline.png")
plt.show()
