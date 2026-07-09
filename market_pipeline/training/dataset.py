"""
Eğitim dataseti oluşturucu.
S&P 500 hisselerinden walk-forward pencereler üretir.
Her pencere: (context fiyat serisi, özellik vektörü, gerçek gelecek getiri)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from config import CONTEXT_LEN, HORIZON_LEN, WALK_STEP, MIN_WINDOWS, DATA_DIR

DATASET_CACHE = DATA_DIR / "training_windows.npz"


def build_windows(
    close_df:    pd.DataFrame,
    feature_dict: dict,          # {sym: DataFrame(date x features)}
    force:       bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tüm hisseler üzerinde kayan pencere oluşturur.

    Döndürür:
        X_price  : (N, CONTEXT_LEN)  — normalize fiyat serisi
        X_feat   : (N, n_features)   — pencere sonu özellik vektörü
        y        : (N,)              — pencere sonrası HORIZON_LEN getirisi
    """
    if not force and DATASET_CACHE.exists():
        print(f"  Dataset onbellekten: {DATASET_CACHE.name}")
        data = np.load(DATASET_CACHE)
        return data["X_price"], data["X_feat"], data["y"]

    # Tüm hisselerdeki union özellik listesi belirlenir
    all_feat_cols = []
    for sym in close_df.columns:
        df = feature_dict.get(sym)
        if df is not None:
            for c in df.columns:
                if c not in all_feat_cols:
                    all_feat_cols.append(c)
    n_features = len(all_feat_cols)
    feat_col_idx = {c: i for i, c in enumerate(all_feat_cols)}

    X_price_list, X_feat_list, y_list, date_ts_list = [], [], [], []
    skipped = 0

    for sym in tqdm(close_df.columns, desc="  Pencere ureti"):
        prices = close_df[sym].dropna().values.astype(np.float32)
        if len(prices) < CONTEXT_LEN + HORIZON_LEN + WALK_STEP:
            skipped += 1
            continue

        feats_df = feature_dict.get(sym)
        if feats_df is None:
            skipped += 1
            continue

        # dropna() yerine indeks olarak price serisinin tarihlerini kullan
        # Herhangi bir özellik sütunundaki NaN zaten feat_vec oluşturmada sıfırlanıyor
        price_index = close_df[sym].dropna().index
        n = len(prices)
        windows_added = 0

        for end in range(CONTEXT_LEN, n - HORIZON_LEN, WALK_STEP):
            ctx = prices[end - CONTEXT_LEN: end]

            # Özellik vektörü
            date_idx = price_index[end - 1]
            if date_idx not in feats_df.index:
                continue
            # Tüm özellik kolonlarını aynı boyuta hizala
            feat_vec = np.zeros(n_features, dtype=np.float32)
            row = feats_df.loc[date_idx]
            for col in feats_df.columns:
                if col in feat_col_idx:
                    v = row[col]
                    fv = 0.0 if (v != v) else float(v)          # NaN → 0
                    if not np.isfinite(fv):                       # inf → 0
                        fv = 0.0
                    feat_vec[feat_col_idx[col]] = fv

            # Hedef: HORIZON_LEN sonraki ortalama getiri
            future = prices[end: end + HORIZON_LEN]
            y_val  = float(np.mean(future) / (ctx[-1] + 1e-9) - 1.0)  # getiri

            # Fiyat serisini normalize et (z-score)
            mu  = float(np.mean(ctx))
            std = float(np.std(ctx)) + 1e-9
            ctx_norm = (ctx - mu) / std

            X_price_list.append(ctx_norm)
            X_feat_list.append(feat_vec)
            y_list.append(y_val)
            date_ts_list.append(pd.Timestamp(date_idx).timestamp())
            windows_added += 1

        if windows_added < MIN_WINDOWS:
            # Az pencereli hisseleri kaldır
            X_price_list  = X_price_list[: -windows_added]
            X_feat_list   = X_feat_list[: -windows_added]
            y_list        = y_list[: -windows_added]
            date_ts_list  = date_ts_list[: -windows_added]
            skipped += 1

    X_price      = np.array(X_price_list,  dtype=np.float32)
    X_feat       = np.array(X_feat_list,   dtype=np.float32)
    y            = np.array(y_list,         dtype=np.float32)
    window_dates = np.array(date_ts_list,   dtype=np.float64)  # Unix timestamp

    feat_cols_arr = np.array(all_feat_cols, dtype=object)
    np.savez_compressed(DATASET_CACHE,
                        X_price=X_price, X_feat=X_feat, y=y,
                        feat_cols=feat_cols_arr, window_dates=window_dates)
    print(f"\n  Toplam pencere: {len(y):,}  |  Atlanan hisse: {skipped}")
    print(f"  X_price: {X_price.shape}  X_feat: {X_feat.shape}")
    return X_price, X_feat, y


def train_val_split(X_price, X_feat, y,
                    val_ratio: float = 0.15):
    """Son %15'i validasyon olarak ayır (zaman sırası korunur)."""
    n     = len(y)
    split = int(n * (1 - val_ratio))
    return (
        X_price[:split], X_feat[:split], y[:split],
        X_price[split:], X_feat[split:], y[split:],
    )
