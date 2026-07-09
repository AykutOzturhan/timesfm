"""
XGBoost piyasa modeli — tüm S&P500 özelliklerini öğrenir.
Görev: 21 günlük yön tahmini — P(bullish) sınıflandırıcı.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import pickle
from config import DATA_DIR, XGB_TREES, XGB_DEPTH, XGB_LR

MODEL_PATH = DATA_DIR / "xgb_market.pkl"


class MarketXGB:
    """
    Piyasa genelinde XGBoost yön sınıflandırıcısı.
    predict() -> (P(bullish) - 0.5): pozitif = bullish sinyal, range: -0.5..+0.5
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.model  = None
        self.feature_names = None

    def fit(self, X_feat: np.ndarray, y: np.ndarray,
            feature_names: list[str] | None = None,
            window_dates: np.ndarray | None = None) -> dict:
        """
        X_feat       : (N, F) — özellik matrisi
        y            : (N,)   — getiri hedefi (float); binary label içeride üretilir
        window_dates : (N,)   — Unix timestamp (float64); yakın tarihli verilere daha yüksek ağırlık
        """
        self.feature_names = feature_names

        Xs    = self.scaler.fit_transform(X_feat)
        y_dir = (y > 0).astype(int)   # 1=bullish, 0=bearish

        # Sample weighting devre dışı — interleaved pencere yapısıyla zarar veriyor.
        # Yarı ömür = 730d ve 365d ikisi de overall accuracy'yi düşürdü (V10, V10b).
        # Pencereler hisse bazında interleaved olduğundan recency ağırlıklandırması
        # yanlış bir iç dağılım oluşturuyor.
        sample_weight = None

        self.model = xgb.XGBClassifier(
            n_estimators     = XGB_TREES,
            max_depth        = XGB_DEPTH,
            learning_rate    = XGB_LR,
            subsample        = 0.8,
            colsample_bytree = 0.7,
            min_child_weight = 5,
            gamma            = 0.1,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            random_state     = 42,
            objective        = "binary:logistic",
            eval_metric      = "logloss",
            tree_method      = "hist",
            n_jobs           = -1,
        )

        # TimeSeriesSplit CV — yön doğruluğunu raporla
        tscv     = TimeSeriesSplit(n_splits=5)
        acc_list = []
        for tr, va in tscv.split(Xs):
            sw_tr = sample_weight[tr] if sample_weight is not None else None
            self.model.fit(Xs[tr], y_dir[tr],
                           sample_weight=sw_tr,
                           eval_set=[(Xs[va], y_dir[va])],
                           verbose=False)
            prob_va  = self.model.predict_proba(Xs[va])[:, 1]
            acc_list.append(float(np.mean((prob_va > 0.5) == y_dir[va])))

        # Final model tüm veriyle
        self.model.fit(Xs, y_dir, sample_weight=sample_weight, verbose=False)

        return {
            "cv_dir_acc_mean": float(np.mean(acc_list)),
            "cv_dir_acc_std":  float(np.std(acc_list)),
            "n_samples":       len(y),
        }

    def predict(self, X_feat: np.ndarray) -> np.ndarray:
        """Returns (P(bullish) - 0.5): pozitif = bullish, negatif = bearish."""
        Xs   = self.scaler.transform(X_feat)
        prob = self.model.predict_proba(Xs)[:, 1]
        return prob - 0.5

    def feature_importance(self, top_n: int = 20) -> pd.Series:
        if self.model is None:
            return pd.Series()
        imp = self.model.feature_importances_
        names = self.feature_names or [f"f{i}" for i in range(len(imp))]
        return pd.Series(imp, index=names).sort_values(ascending=False).head(top_n)

    def direction_accuracy(self, X_feat: np.ndarray, y: np.ndarray) -> float:
        """Yön doğruluğu: P(bullish) > 0.5 ise bullish tahmin, y > 0 ise bullish gerçek."""
        pred = self.predict(X_feat)   # prob - 0.5; sign = tahmini yön
        return float(np.mean(np.sign(pred) == np.sign(y)))

    def save(self, path=MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  XGB modeli kaydedildi: {path}")

    @classmethod
    def load(cls, path=MODEL_PATH) -> "MarketXGB":
        with open(path, "rb") as f:
            return pickle.load(f)
