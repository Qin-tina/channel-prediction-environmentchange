import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Load training and testing data
train_snr = pd.read_csv("train_snr.csv")["snr_db"].values
test_snr = pd.read_csv("test_snr.csv")["snr_db"].values

# Step 2: Prepare training data
# Use previous SNR to predict next SNR
X_train = train_snr[:-1].reshape(-1, 1)
y_train = train_snr[1:]

# Step 3: Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Prepare testing data
X_test = test_snr[:-1].reshape(-1, 1)
y_test = test_snr[1:]

# Step 5: Last-value Prediction
last_value_pred = X_test.flatten()

# Step 6: Linear Regression Prediction
linear_pred = model.predict(X_test)

# Step 7: Calculate RMSE
last_value_rmse = np.sqrt(
    np.mean((y_test - last_value_pred) ** 2)
)

linear_rmse = np.sqrt(
    np.mean((y_test - linear_pred) ** 2)
)

# Step 8: Display results
print("Training and Testing Completed!")
print("Training samples:", len(X_train))
print("Testing predictions:", len(X_test))

print("\nModel Parameters:")
print("Coefficient:", round(model.coef_[0], 4))
print("Intercept:", round(model.intercept_, 4))

print("\nTest Set RMSE Comparison:")
print(f"Last-value RMSE: {last_value_rmse:.4f} dB")
print(f"Linear Regression RMSE: {linear_rmse:.4f} dB")
# Step 11: Save prediction results

results = pd.DataFrame({
    "time_step": np.arange(1, len(y_test) + 1),
    "actual_snr": y_test,
    "last_value_prediction": last_value_pred,
    "linear_prediction": linear_pred
})

results.to_csv("prediction_results.csv", index=False)

print("\nPrediction results saved successfully!")
print(results.head())
# Step 12: Plot test predictions

import matplotlib.pyplot as plt

# Display the first 100 test predictions
n = 100

plt.figure(figsize=(12, 5))

plt.plot(
    results["time_step"][:n],
    results["actual_snr"][:n],
    label="Actual SNR",
    color="black",
    linewidth=2
)

plt.plot(
    results["time_step"][:n],
    results["last_value_prediction"][:n],
    label="Last-value Prediction",
    color="blue",
    linestyle="--"
)

plt.plot(
    results["time_step"][:n],
    results["linear_prediction"][:n],
    label="Linear Regression",
    color="red",
    linestyle=":"
)

plt.xlabel("Time Step")
plt.ylabel("SNR (dB)")
plt.title("Channel Prediction on Test Data")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
plt.savefig(
    "test_prediction_comparison.png",
    dpi=300
)

plt.show()
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