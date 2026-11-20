
import pandas as pd
import numpy as np

# Load prediction results
data = pd.read_csv("environment_B_predictions.csv")

print("Prediction Data Check")
print("-" * 40)

# Check number of rows
print("Number of rows:", len(data))

# Check column names
print("\nColumns:")
print(data.columns.tolist())

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Show first five rows
print("\nFirst 5 rows:")
print(data.head())

# Calculate RMSE from saved predictions
actual = data["actual_snr"].values
last_pred = data["last_value_prediction"].values
linear_pred = data["linear_prediction"].values

last_rmse = np.sqrt(np.mean((actual - last_pred) ** 2))
linear_rmse = np.sqrt(np.mean((actual - linear_pred) ** 2))

print("\nRMSE Verification:")
print(f"Last-value RMSE: {last_rmse:.4f} dB")
print(f"Linear Regression RMSE: {linear_rmse:.4f} dB")

# Final check
required_columns = [
    "time_step",
    "actual_snr",
    "last_value_prediction",
    "linear_prediction"
]

if (
    all(col in data.columns for col in required_columns)
    and not data.isnull().any().any()
    and len(data) > 0
):
    print("\nDATA CHECK PASSED!")
else:
    print("\nDATA CHECK FAILED!")
