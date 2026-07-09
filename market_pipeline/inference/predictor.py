"""
Tahmin motoru — herhangi bir ABD hissesi için tahmin üretir.
TimesFM base forecast + XGBoost düzeltmesi.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
import timesfm
from timesfm import configs
from datetime import timedelta
from config import CONTEXT_LEN, HORIZON_LEN, DATA_DIR
from features.technical import compute_all
from features.macro_features import build_macro_features
from features.sector import build_sector_features, _sector_to_etf
from training.xgb_market import MarketXGB


class MarketPredictor:
    """
    Tek arayüz: herhangi bir hisse için tam pipeline tahmini.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._tfm   = None
        self._xgb   = None

    def _load_tfm(self):
        if self._tfm is None:
            print("  TimesFM yukleniyor...")
            self._tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch", device=self.device
            )
            self._tfm.compile(configs.ForecastConfig(
                max_context=512, max_horizon=HORIZON_LEN, per_core_batch_size=8
            ))

    def _load_xgb(self):
        if self._xgb is None:
            try:
                self._xgb = MarketXGB.load()
                print("  XGB modeli yuklendi.")
            except FileNotFoundError:
                print("  XGB modeli bulunamadi (once train_pipeline.py calistirin)")

    def predict(self, symbol: str, sector: str = "",
                horizon: int = HORIZON_LEN) -> dict:
        """
        symbol  : hisse kodu (ör. "AAPL")
        sector  : GICS sektörü (boş bırakılırsa otomatik)
        horizon : kaç iş günü ileriye (default 21 = ~1 ay)
        """
        self._load_tfm()
        self._load_xgb()

        # ── 1. Veri Çek ───────────────────────────────────────────────
        print(f"  [{symbol}] veri cekiliyor...")
        raw = yf.download(symbol, period="3y", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"{symbol} verisi bulunamadi")

        close  = raw["Close"].squeeze().dropna()
        volume = raw["Volume"].squeeze().dropna()
        high   = raw["High"].squeeze().dropna()
        low    = raw["Low"].squeeze().dropna()

        prices     = close.values.astype(np.float32)[-CONTEXT_LEN:]
        dates      = close.index[-CONTEXT_LEN:]
        current    = float(prices[-1])

        # ── 2. Teknik Özellikler ──────────────────────────────────────
        close_df  = close.to_frame().T   # (1, n_dates)
        vol_df    = volume.to_frame().T
        high_df   = high.to_frame().T
        low_df    = low.to_frame().T
        # Transpoze et: (n_dates, 1)
        close_df  = close.to_frame(symbol)
        vol_df    = volume.to_frame(symbol)
        high_df   = high.to_frame(symbol)
        low_df    = low.to_frame(symbol)

        tech = compute_all(close_df, high_df, low_df, vol_df)
        feat_frames = {name: df[symbol] for name, df in tech.items()}
        feat_df = pd.DataFrame(feat_frames)

        # Makro verisi
        macro_syms = ["SPY", "^VIX", "TLT", "UUP", "^TNX"]
        macro_raw  = yf.download(macro_syms, period="3y", interval="1d",
                                  auto_adjust=True, progress=False)
        macro_df = pd.DataFrame()
        for sym2 in macro_syms:
            col = sym2.replace("^", "")
            try:
                s = macro_raw["Close"][sym2].squeeze().dropna()
                s.name = col
                macro_df[col] = s
            except Exception:
                pass

        macro_feats = build_macro_features(macro_df, close_df)
        feat_df = feat_df.join(macro_feats, how="left")

        # Sektör göreli güç
        etf_sym = _sector_to_etf(sector)
        if etf_sym:
            etf_raw = yf.download(etf_sym, period="3y", interval="1d",
                                   auto_adjust=True, progress=False)
            etf_close = etf_raw["Close"].squeeze().dropna()
            stk_ret20 = close.pct_change(20)
            etf_ret20 = etf_close.pct_change(20)
            feat_df["RS_sector_20d"] = stk_ret20 - etf_ret20

        feat_df = feat_df.ffill().fillna(0.0)

        # ── 3. TimesFM Base Forecast ───────────────────────────────────
        point_fc, quant_fc = self._tfm.forecast(
            horizon=horizon, inputs=[prices]
        )
        base_point  = point_fc[0]
        base_quants = quant_fc[0]

        # ── 4. XGBoost Düzeltme ────────────────────────────────────────
        correction = 0.0
        if self._xgb is not None and self._xgb.feature_names is not None:
            try:
                # Modelin beklediği özellik isimlerine hizala
                expected = self._xgb.feature_names
                aligned  = np.zeros(len(expected), dtype=np.float32)
                last_row = feat_df.iloc[-1]
                for i, col in enumerate(expected):
                    if col in last_row.index:
                        v = last_row[col]
                        aligned[i] = 0.0 if (v != v) else float(v)
                correction = float(self._xgb.predict(aligned.reshape(1, -1))[0])
                correction = correction * current
            except Exception as e:
                print(f"  XGB duzeltme hatasi: {e}")

        corrected_point = base_point + correction

        # ── 5. Sonuç ─────────────────────────────────────────────────
        last_date = dates[-1]
        biz_days  = pd.bdate_range(
            start=last_date + timedelta(days=1), periods=horizon
        )

        milestones = {}
        for label, idx in [("2W", 9), ("1M", 20), ("2M", 41), ("3M", 62)]:
            if idx < horizon:
                p    = float(corrected_point[idx])
                chg  = (p - current) / current * 100
                milestones[label] = {"price": p, "change_pct": chg}

        return {
            "symbol":          symbol,
            "current":         current,
            "base_forecast":   base_point.tolist(),
            "corrected":       corrected_point.tolist(),
            "quantiles":       base_quants.tolist(),
            "correction_usd":  correction,
            "biz_days":        biz_days,
            "milestones":      milestones,
            "features":        feat_df,
        }
