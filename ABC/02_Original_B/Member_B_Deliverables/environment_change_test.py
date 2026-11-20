import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load Environment A training data
train_snr = pd.read_csv("train_snr.csv")["snr_db"].values

# Prepare training samples
X_train = train_snr[:-1].reshape(-1, 1)
y_train = train_snr[1:]

# Train model only on Environment A
model = LinearRegression()
model.fit(X_train, y_train)

# Generate Environment B with lower correlation
rng = np.random.default_rng(123)

n_samples = 1000
rho_B = 0.60

snr_B = np.zeros(n_samples)
snr_B[0] = 10.0

for t in range(1, n_samples):
    snr_B[t] = (
        10
        + rho_B * (snr_B[t - 1] - 10)
        + np.sqrt(1 - rho_B**2) * rng.normal(0, 2)
    )

# Prepare Environment B test data
X_test = snr_B[:-1].reshape(-1, 1)
y_test = snr_B[1:]

# Predict using both methods
last_value_pred = X_test.flatten()
linear_pred = model.predict(X_test)

# Calculate RMSE
last_value_rmse = np.sqrt(
    np.mean((y_test - last_value_pred) ** 2)
)

linear_rmse = np.sqrt(
    np.mean((y_test - linear_pred) ** 2)
)

# Display results
print("Environment Change Experiment")
print("Training Environment A: rho = 0.95")
print("Testing Environment B: rho = 0.60")

print("\nRMSE after Environment Change:")
print(f"Last-value RMSE: {last_value_rmse:.4f} dB")
print(f"Linear Regression RMSE: {linear_rmse:.4f} dB")

# Save predictions for the other group members
results = pd.DataFrame({
    "time_step": np.arange(1, len(y_test) + 1),
    "actual_snr": y_test,
    "last_value_prediction": last_value_pred,
    "linear_prediction": linear_pred
})

results.to_csv("environment_B_predictions.csv", index=False)

print("\nEnvironment B predictions saved successfully!")