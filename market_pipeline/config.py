"""Merkezi konfigürasyon — tüm modüller buradan okur."""
from pathlib import Path

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Veri Ayarları ─────────────────────────────────────────────────────────
PERIOD      = "7y"          # yfinance indirme penceresi (COVID 2020 dahil)
INTERVAL    = "1d"          # günlük veri
MIN_HISTORY = 600           # bu kadar günden az olan hisseler atlanır

# ── Özellik Ayarları ──────────────────────────────────────────────────────
RSI_PERIOD   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
BB_PERIOD    = 20
ATR_PERIOD   = 14
ADX_PERIOD   = 14
OBV_EMA      = 20
CORR_WINDOW  = 60           # sektör korelasyon penceresi

# ── Eğitim Ayarları ───────────────────────────────────────────────────────
CONTEXT_LEN  = 252          # ~1 yıl geçmiş
HORIZON_LEN  = 21           # ~1 ay ileriye
WALK_STEP    = 5            # walk-forward kaydırma adımı (küçük = daha fazla pencere)
MIN_WINDOWS  = 10           # bir hisseden en az bu kadar pencere lazım

LORA_RANK    = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LR           = 1e-4
EPOCHS       = 5            # CPU için tutumlu
BATCH_SIZE   = 16

XGB_TREES    = 500
XGB_DEPTH    = 5
XGB_LR       = 0.03

# ── Makro Tickerlar ───────────────────────────────────────────────────────
MACRO_TICKERS = {
    "SPY":  "S&P 500 ETF",
    "QQQ":  "Nasdaq ETF",
    "IWM":  "Russell 2000",
    "^VIX": "VIX",
    "TLT":  "20Y Treasury",
    "^TNX": "10Y Yield",
    "^IRX": "3M Yield",
    "SHY":  "1-3Y Treasury (Fed beklenti proxy)",
    "UUP":  "USD Index",
    "GLD":  "Gold",
    "USO":  "Oil",
    "TIP":  "TIPS (enflasyon beklentisi)",
    "HYG":  "High Yield Bond (kredi riski iştahı)",
    "LQD":  "Investment Grade Bond",
    "XAR":  "Aerospace & Defense ETF (gov funding proxy)",
    "ITB":  "US Home Construction (infrastructure proxy)",
    "^VVIX": "VIX of VIX (options sentiment volatilitesi)",
    "^VIX3M": "3 Aylik VIX (orta vadeli belirsizlik)",
    "^SKEW": "CBOE SKEW (kuyruk riski / put premium)",
}

# ── API Anahtarları ───────────────────────────────────────────────────────
# Anahtar repoya girmez: POLYGON_API_KEY ortam değişkeninden ya da
# gitignore'lu config_local.py dosyasından okunur.
import os
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
try:
    from config_local import POLYGON_API_KEY  # noqa: F401
except ImportError:
    pass

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication",
}
