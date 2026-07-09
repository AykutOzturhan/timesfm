"""Teknik indikatörler — tüm S&P 500 için vektörize hesaplama."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, ATR_PERIOD, ADX_PERIOD


def _rsi(close: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs))


def _macd_histogram(close: pd.DataFrame) -> pd.DataFrame:
    ema_f = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd - sig   # histogram: pozitif = bullish momentum


def _bb_pct(close: pd.DataFrame, period: int = BB_PERIOD) -> pd.DataFrame:
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return (close - lower) / (upper - lower + 1e-9)


def _atr_norm(high: pd.DataFrame, low: pd.DataFrame,
              close: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """ATR / close → normalize volatilite."""
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=0).groupby(level=0).max()       # her tarih için max
    atr = tr.rolling(period).mean()
    return atr / (close + 1e-9)


def _adx(high: pd.DataFrame, low: pd.DataFrame,
         close: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Basitleştirilmiş ADX (trend gücü 0-100)."""
    up   = high.diff()
    down = -low.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=0).groupby(level=0).max()
    atr14    = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / (atr14 + 1e-9)
    dx  = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100
    adx = dx.ewm(com=period - 1, adjust=False).mean()
    return adx


def _volume_zscore(volume: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    mu  = volume.rolling(period).mean()
    std = volume.rolling(period).std()
    return (volume - mu) / (std + 1e-9)


def _obv_trend(close: pd.DataFrame, volume: pd.DataFrame,
               period: int = 20) -> pd.DataFrame:
    """OBV'nin 20 günlük değişim trendi (normalize)."""
    direction = close.diff().apply(np.sign)
    obv = (direction * volume).cumsum()
    return obv.pct_change(period)


def _stochastic(high: pd.DataFrame, low: pd.DataFrame,
                close: pd.DataFrame, k: int = 14) -> pd.DataFrame:
    low_k  = low.rolling(k).min()
    high_k = high.rolling(k).max()
    return (close - low_k) / (high_k - low_k + 1e-9) * 100


def _distance_from_high(close: pd.DataFrame, period: int = 52) -> pd.DataFrame:
    """52 haftalık zirveden uzaklık (yüzde)."""
    weeks = period * 5
    peak  = close.rolling(weeks).max()
    return (close - peak) / (peak + 1e-9)


def _ma_crossover(close: pd.DataFrame,
                  fast: int = 50, slow: int = 200):
    """
    Golden cross / death cross tespiti.
    MA_cross: +1 = MA50 > MA200 (yükselen trend), -1 = ters.
    Price_vs_MA50/200: fiyatın hareketli ortalamaya uzaklığı.
    """
    ma_fast = close.rolling(fast, min_periods=fast).mean()
    ma_slow = close.rolling(slow, min_periods=slow).mean()
    cross   = np.sign(ma_fast - ma_slow)
    vs_50   = (close / (ma_fast + 1e-9) - 1).clip(-0.30, 0.30)
    vs_200  = (close / (ma_slow + 1e-9) - 1).clip(-0.50, 0.50)
    return cross, vs_50, vs_200


def _trend_consistency(close: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Son N günde pozitif kapanış oranı (0..1). 0.5+ = baskın yükseliş."""
    return (close.diff() > 0).rolling(period).mean()


def _price_channel_pct(close: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Fiyatın son N günlük kanal içindeki konumu (0=dip, 1=tepe)."""
    hi = close.rolling(period).max()
    lo = close.rolling(period).min()
    return (close - lo) / (hi - lo + 1e-9)


def compute_all(close: pd.DataFrame,
                high: pd.DataFrame,
                low:  pd.DataFrame,
                volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Tüm teknik indikatörleri hesaplar.
    Döndürür: {isim: DataFrame(tarih x hisse)} sözlüğü
    """
    ret_1  = close.pct_change(1)
    ret_5  = close.pct_change(5)
    ret_20 = close.pct_change(20)
    ret_60 = close.pct_change(60)

    ma_cross, vs_ma50, vs_ma200 = _ma_crossover(close)

    return {
        "RSI":            _rsi(close),
        "MACD_hist":      _macd_histogram(close),
        "BB_pct":         _bb_pct(close),
        "ATR_norm":       _atr_norm(high, low, close),
        "ADX":            _adx(high, low, close),
        "Stoch":          _stochastic(high, low, close),
        "Vol_Z":          _volume_zscore(volume),
        "OBV_trend":      _obv_trend(close, volume),
        "Ret_1d":         ret_1,
        "Ret_3d":         close.pct_change(3),          # çok kısa vadeli momentum
        "Ret_5d":         ret_5,
        "Ret_20d":        ret_20,
        "Ret_60d":        ret_60,
        "DistHigh52W":    _distance_from_high(close),
        "MA_cross":       ma_cross,
        "Price_vs_MA50":  vs_ma50,
        "Price_vs_MA200": vs_ma200,
        "TrendStr_5d":    _trend_consistency(close, 5),  # 5 günlük trend tutarlılığı
        "TrendStr_20d":   _trend_consistency(close, 20),
        "PriceChannel":   _price_channel_pct(close, 20),
        "PriceChannel5d": _price_channel_pct(close, 5),  # 5-günlük kanal konumu
    }
