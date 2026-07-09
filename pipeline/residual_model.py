"""
Aşama 4 — XGBoost artık düzeltme modeli.

TimesFM bir base forecast üretir.
Bu modül, TimesFM'in geçmişteki artıklarını (hata = gerçek - tahmin)
tüm özelliklerle (RSI, MACD, sentiment, macro...) öğrenerek
gelecekteki tahmini düzeltir.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("xgboost kurulu degil: pip install xgboost")


def make_residual_dataset(
    features_df: pd.DataFrame,
    uec_prices: pd.Series,
    tfm_forecast_fn,
    horizon: int = 21,
    context: int = 252,
    step: int = 5,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Kayan pencere yöntemiyle (walk-forward) TimesFM tahminleri üretir
    ve gerçek değerlerden farkı (artık) hesaplar.

    features_df : özellik tablosu (tarih indeksli)
    uec_prices  : UEC kapanış fiyatları
    tfm_forecast_fn : (prices_array) -> point_forecast_array fonksiyonu
    horizon     : kaç gün ileriye tahmin (21 = ~1 ay)
    context     : TimesFM'e verilen geçmiş uzunluğu
    step        : pencere kaydırma adımı

    Döndürür:
        X : (n_samples, n_features) — artık tahmini için özellikler
        y : (n_samples,) — gerçek artık (residual)
        dates : tahmin tarihlerinin listesi
    """
    aligned = features_df.join(uec_prices.rename("Price"), how="inner").dropna()
    prices = aligned["Price"].values
    feat_cols = [c for c in aligned.columns if c != "Price"]
    feats = aligned[feat_cols].values

    X_list, y_list, dates = [], [], []
    n = len(prices)

    for end in range(context, n - horizon, step):
        ctx_prices = prices[end - context: end]
        # TimesFM tahmini (yalnızca 1 adım: horizon sonraki ortalama)
        try:
            fc = tfm_forecast_fn(ctx_prices)  # (horizon,) array
            tfm_pred = float(np.mean(fc[:horizon]))
        except Exception:
            continue

        actual = float(np.mean(prices[end: end + horizon]))
        residual = actual - tfm_pred

        # Tahmin anındaki özellik vektörü
        feat_vec = feats[end - 1]
        X_list.append(feat_vec)
        y_list.append(residual)
        dates.append(aligned.index[end])

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(y_list), dates


class ResidualCorrector:
    """XGBoost tabanlı artık düzeltme modeli."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 4,
                 lr: float = 0.05):
        self.scaler = StandardScaler()
        self.model  = None
        self.params  = dict(n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=lr, subsample=0.8,
                            colsample_bytree=0.8, random_state=42,
                            objective="reg:squarederror")

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not XGB_OK:
            raise RuntimeError("xgboost kurulu degil")
        Xs = self.scaler.fit_transform(X)
        self.model = xgb.XGBRegressor(**self.params)

        # TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores = []
        for tr_idx, va_idx in tscv.split(Xs):
            self.model.fit(Xs[tr_idx], y[tr_idx],
                           eval_set=[(Xs[va_idx], y[va_idx])],
                           verbose=False)
            pred = self.model.predict(Xs[va_idx])
            mae_scores.append(mean_absolute_error(y[va_idx], pred))

        # Son modeli tüm veriyle eğit
        self.model.fit(Xs, y, verbose=False)
        return {"cv_mae_mean": float(np.mean(mae_scores)),
                "cv_mae_std":  float(np.std(mae_scores))}

    def predict(self, feat_vec: np.ndarray) -> float:
        """Tek bir özellik vektörü için artık tahmini."""
        if self.model is None:
            return 0.0
        x = self.scaler.transform(feat_vec.reshape(1, -1))
        return float(self.model.predict(x)[0])

    def feature_importance(self, feature_names: list[str]) -> pd.Series:
        if self.model is None:
            return pd.Series()
        imp = self.model.feature_importances_
        return pd.Series(imp, index=feature_names).sort_values(ascending=False)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ResidualCorrector":
        with open(path, "rb") as f:
            return pickle.load(f)
