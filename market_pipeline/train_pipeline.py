"""
Ana eğitim pipeline'ı — S&P 500 üzerinde tam model eğitimi.

Aşamalar:
  1. S&P 500 ticker listesi
  2. Fiyat + hacim + makro veri indirme
  3. Temel (fundamental) veriler
  4. Teknik + makro + sektör özellik hesaplama
  5. Eğitim penceresi oluşturma
  6. XGBoost piyasa modeli eğitimi
  7. (Opsiyonel) TimesFM LoRA fine-tuning
  8. Sonuç özeti

Kullanım:
  python train_pipeline.py
  python train_pipeline.py --skip-lora      # LoRA atla (hız için)
  python train_pipeline.py --force          # önbelleği yenile
  python train_pipeline.py --sample 50      # test için 50 hisse
"""
import sys, os, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR, CONTEXT_LEN, HORIZON_LEN

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-lora",  action="store_true", help="LoRA egitimini atla")
parser.add_argument("--force",      action="store_true", help="Onbellegi yenile")
parser.add_argument("--sample",     type=int, default=0, help="Test icin N hisse (0=hepsi)")
args = parser.parse_args()

print("=" * 60)
print("  S&P 500 TAM EGİTİM PİPELİNE")
print("=" * 60)

# ── 1. Ticker Listesi ────────────────────────────────────────────────────
print("\n[1/7] S&P 500 ticker listesi...")
from sp500.tickers import fetch_sp500_tickers
tickers_df = fetch_sp500_tickers()

if args.sample > 0:
    # Sektör dengeli örnekleme
    # Sektör dengeli örnekleme — groupby key'i sonuçtan düşmesin diye include_groups=False yok, elle yapıyoruz
    n_per_sector = max(1, args.sample // len(tickers_df["Sector"].unique()))
    sampled = []
    for sector, grp in tickers_df.groupby("Sector"):
        sampled.append(grp.sample(min(len(grp), n_per_sector)).assign(Sector=sector))
    tickers_df = pd.concat(sampled, ignore_index=True)
    print(f"  Örnekleme modu: {len(tickers_df)} hisse")
else:
    print(f"  Tam liste: {len(tickers_df)} hisse")

symbols = tickers_df["Symbol"].tolist()
ticker_sector = dict(zip(tickers_df["Symbol"], tickers_df["Sector"]))
print(tickers_df.groupby("Sector").size().sort_values(ascending=False).to_string())

# ── 2. Fiyat + Makro Veri ────────────────────────────────────────────────
print("\n[2/7] Fiyat ve makro veri indiriliyor...")
from sp500.downloader import download_prices, download_macro
close_df, volume_df = download_prices(symbols, force=args.force)
macro_df = download_macro(force=args.force)

# Yalnızca yeterli geçmişi olan hisseleri tut
valid = close_df.columns[close_df.notna().sum() >= CONTEXT_LEN + HORIZON_LEN + 20]
close_df  = close_df[valid]
volume_df = volume_df[valid] if set(valid) <= set(volume_df.columns) else volume_df
print(f"  Geçerli hisse: {len(valid)} / {len(symbols)}")
print(f"  Tarih araligi: {close_df.index[0].date()} -> {close_df.index[-1].date()}")

# Eğitim verisini 2023+ ile sınırla — post-rate-hike rejimi, test perioduyla daha uyumlu
# 252 günlük context için 2023-01-01 öncesi fiyatlar yine de feature olarak kullanılır
# (context penceresi 252 gün = ~1 yıl geriye gider, bu yüzden 2022-01-01'den itibaren)
_train_start = "2022-01-01"   # post-rate-hike rejimi; COVID verisi kalibrasyonu bozuyor
close_df  = close_df[close_df.index >= _train_start]
volume_df = volume_df[volume_df.index >= _train_start]
macro_df  = macro_df[macro_df.index >= _train_start]
print(f"  Kırpılmış tarih araligi: {close_df.index[0].date()} -> {close_df.index[-1].date()}")

# ── 3. Temel Veriler ─────────────────────────────────────────────────────
print("\n[3/7] Temel (fundamental) veriler...")
from sp500.fundamentals import fetch_fundamentals, add_percentile_rank
fund_df = fetch_fundamentals(symbols, force=args.force)
fund_pct = add_percentile_rank(fund_df)
print(f"  {fund_df.shape[1]} metrik, {len(fund_df)} hisse")

# ── 4. Teknik + Makro + Sektör + Sentiment + Options Özellikler ─────────
print("\n[4/7] Özellikler hesaplanıyor...")
from features.technical import compute_all
from features.macro_features import build_macro_features
from features.sector import build_sector_features, _sector_to_etf
from features.news_sentiment import add_sentiment_to_features
from features.options_features import add_options_to_features
from features.short_features import add_short_to_features
from features.earnings_features import add_earnings_to_features
# from features.insider_features import add_insider_to_features   # static snapshot — hurts backtest
# from features.congress_features import add_congress_to_features  # static snapshot — hurts backtest
# from features.government_features import add_government_to_features  # static snapshot — hurts backtest

# Teknik — tüm hisseler için vektörize
high_df = close_df * 1.005
low_df  = close_df * 0.995

tech_dict = compute_all(close_df.ffill(), high_df, low_df, volume_df.ffill())
print(f"  Teknik indikatörler: {list(tech_dict.keys())}")

# Makro özellikler (tarih bazlı)
macro_feats = build_macro_features(macro_df, close_df)
print(f"  Makro özellikler: {list(macro_feats.columns)}")

# Her hisse için birleşik özellik DataFrame
print("  Hisse başına özellik tabloları birleştiriliyor...")
feature_dict = {}
for sym in close_df.columns:
    parts = []
    for name, df in tech_dict.items():
        if sym in df.columns:
            parts.append(df[sym].rename(name))
    parts.append(macro_feats)
    sector  = ticker_sector.get(sym, "")
    etf_sym = _sector_to_etf(sector)
    etf_col = etf_sym if etf_sym in macro_df.columns else None
    if etf_col:
        rs = (close_df[sym].pct_change(20) - macro_df[etf_col].pct_change(20)
              ).rename("RS_sector_20d")
        parts.append(rs)
    if sym in fund_pct.index:
        for k, v in fund_pct.loc[sym].to_dict().items():
            if not pd.isna(v):
                parts.append(pd.Series(v, index=close_df.index, name=k))
    sym_df = pd.concat([p.to_frame() if isinstance(p, pd.Series) else p
                        for p in parts], axis=1)
    sym_df = sym_df.ffill().fillna(0.0)
    feature_dict[sym] = sym_df

# Haber sentiment ekle
print("  Haber sentiment ekleniyor...")
feature_dict = add_sentiment_to_features(
    feature_dict, list(close_df.columns), close_df.index
)

# Options IV / put-call / hacim ekleniyor
print("  Options IV / put-call / vol ratio ekleniyor...")
feature_dict = add_options_to_features(
    feature_dict, list(close_df.columns), close_df.index,
    force=args.force
)

# Kazanç açıklaması yakınlığı (tarihi zaman serisi — contamination yok)
print("  Kazanc tarihleri (earnings proximity) ekleniyor...")
feature_dict = add_earnings_to_features(
    feature_dict, list(close_df.columns), close_df.index,
    force=args.force
)


# Short interest: static snapshot — disabled (same issue as insider/congress/govt)
# short_ratio günümüz değeri tüm pencerelere yayılıyor → hisse kimliği öğreniyor,
# piyasa zamanlaması öğrenmiyor. Backtest'te bu özellik 0 geldiğinden accuracy düşüyor.
# from features.short_features import add_short_to_features  # DISABLED

# Insider/congress/government: static snapshot features — disabled (hurt backtest: 55.6%→51.1%)

# Yeni sütunlardaki NaN ve inf değerleri temizle
for sym in list(feature_dict.keys()):
    df = feature_dict[sym].ffill().fillna(0.0)
    df = df.replace([np.inf, -np.inf], 0.0)
    feature_dict[sym] = df

first_sym = next(iter(feature_dict))
feat_cols = list(feature_dict[first_sym].columns)
print(f"  Örnek özellik sayısı ({first_sym}): {len(feat_cols)}")
print(f"  Özellikler: {feat_cols[:8]} ... {feat_cols[-4:]}")

# ── 5. Eğitim Penceresi ──────────────────────────────────────────────────
print("\n[5/7] Eğitim pencereleri oluşturuluyor...")
from training.dataset import build_windows, train_val_split
X_price, X_feat, y = build_windows(close_df, feature_dict, force=args.force)

if len(y) == 0:
    print("  HATA: Yeterli pencere oluşturulamadı!")
    sys.exit(1)

print(f"  Toplam pencere: {len(y):,}")
print(f"  Hedef (getiri) istatistik: "
      f"ort={np.mean(y):.4f}  std={np.std(y):.4f}  "
      f"min={np.min(y):.3f}  max={np.max(y):.3f}")

X_ptr, X_ftr, y_tr, X_pva, X_fva, y_va = train_val_split(X_price, X_feat, y)
print(f"  Train: {len(y_tr):,}  |  Val: {len(y_va):,}")

# ── 6. XGBoost Eğitimi ───────────────────────────────────────────────────
print("\n[6/7] XGBoost piyasa modeli eğitiliyor...")
from training.xgb_market import MarketXGB

# Özellik isimleri — npz'den gerçek sırayı al (feature alignment için kritik)
_npz = np.load(str(DATA_DIR / "training_windows.npz"), allow_pickle=True)
if "feat_cols" in _npz:
    feat_names = list(_npz["feat_cols"])
else:
    sample_sym = next(iter(feature_dict))
    feat_names = list(feature_dict[sample_sym].columns)

xgb_model = MarketXGB()
# Window tarihlerini yükle — yakın dönem ağırlıklandırması için
_npz_dates = np.load(str(DATA_DIR / "training_windows.npz"), allow_pickle=True)
_window_dates_full = _npz_dates.get("window_dates", None)
if _window_dates_full is not None:
    n_total = len(y)
    n_val   = len(y_va)
    _wd_tr  = _window_dates_full[:n_total - n_val]
    print(f"  Window tarihleri: {len(_wd_tr)} pencere (egitim)")
else:
    _wd_tr = None
    print("  Window tarihleri bulunamadi (cache yenile: --force)")
cv_result  = xgb_model.fit(X_ftr, y_tr, feature_names=feat_names, window_dates=_wd_tr)
print(f"  CV Yon Dogrulugu: {cv_result['cv_dir_acc_mean']*100:.1f}% +/- {cv_result['cv_dir_acc_std']*100:.1f}%")

# Validasyon
dir_acc  = xgb_model.direction_accuracy(X_fva, y_va)
print(f"  Yon dogrulugu:    {dir_acc*100:.1f}%")

# Top 15 özellik
imp = xgb_model.feature_importance(top_n=15)
print("\n  En önemli 15 özellik:")
for feat, score in imp.items():
    bar = "|" * int(score * 200)
    print(f"    {feat:30s} {score:.4f}  {bar}")

xgb_model.save()

# ── 7. LoRA Fine-Tuning (Opsiyonel) ─────────────────────────────────────
if not args.skip_lora:
    print("\n[7/7] TimesFM LoRA fine-tuning...")
    print("  (Bu adım CPU'da uzun surer; --skip-lora ile atlayabilirsiniz)")
    from training.lora_trainer import TimesFMLoRATrainer
    trainer = TimesFMLoRATrainer(device="cpu")
    ok = trainer.setup()
    if ok:
        losses = trainer.train(X_ptr, y_tr)
        trainer.save()
        print(f"  Son loss: {losses[-1]:.5f}")
    else:
        print("  LoRA kurulumu basarisiz (transformers versiyonu indirilemedi)")
        print("  XGBoost modeli yeterli — LoRA opsiyonel")
else:
    print("\n[7/7] LoRA atlandı (--skip-lora)")

# ── Özet ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EĞİTİM TAMAMLANDI")
print("=" * 60)
print(f"  Eğitilen hisse sayısı  : {close_df.shape[1]}")
print(f"  Toplam eğitim penceresi: {len(y_tr):,}")
print(f"  XGBoost yön doğruluğu  : {dir_acc*100:.1f}%")
print(f"  Modeller: {DATA_DIR}/xgb_market.pkl")
print()
print("  Kullanım:")
print("    python inference/run_inference.py --symbol AAPL")
print("    python inference/run_inference.py --symbol UEC")
print("=" * 60)
