import numpy as np
import pandas as pd

# Make results reproducible
rng = np.random.default_rng(42)

# Generate 1000 correlated SNR samples
n_samples = 1000
rho = 0.95

snr = np.zeros(n_samples)
snr[0] = 10.0

for t in range(1, n_samples):
    snr[t] = (
        10
        + rho * (snr[t - 1] - 10)
        + np.sqrt(1 - rho**2) * rng.normal(0, 2)
    )

# Split chronologically: 70% training, 30% testing
train_snr = snr[:700]
test_snr = snr[700:]

# Save CSV files
pd.DataFrame({"snr_db": train_snr}).to_csv(
    "train_snr.csv", index=False
)

pd.DataFrame({"snr_db": test_snr}).to_csv(
    "test_snr.csv", index=False
)

print("Data preparation completed!")
print("Training samples:", len(train_snr))
print("Testing samples:", len(test_snr))
print("First 5 training SNR values:")
print(train_snr[:5])