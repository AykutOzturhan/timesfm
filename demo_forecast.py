"""TimesFM hızlı demo - sinüs dalgası üzerinde tahmin."""
import numpy as np
import timesfm
from timesfm import configs

print("TimesFM model yukleniyor...")

tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    device="cpu",
)

# Model derleme
forecast_config = configs.ForecastConfig(
    max_context=128,
    max_horizon=24,
    per_core_batch_size=8,
)
tfm.compile(forecast_config)
print("Model derlendi.")

# Gerçekçi örnek: 100 noktalı sinüs dalgası + gürültü
np.random.seed(42)
t = np.arange(100)
series = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.randn(100)

print("\nGiris serisi uzunlugu:", len(series))
print("Ortalama:", round(float(np.mean(series)), 4))

# Tahmin yap (24 adım ileriye)
point_forecast, quantile_forecast = tfm.forecast(
    horizon=24,
    inputs=[series],
)

print("\n24 adimlik nokta tahmini:")
for i, v in enumerate(point_forecast[0]):
    print(f"  t+{i+1:2d}: {v:.4f}")

print("\nTamamlandi!")
