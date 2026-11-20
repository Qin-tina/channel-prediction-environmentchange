
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load("fixed_linear_model.pkl")

# Load Environment B predictions
data = pd.read_csv("environment_B_predictions.csv")

# Previous SNR measurement is the model input
X = data["last_value_prediction"].values.reshape(-1, 1)

# Predict without retraining
new_predictions = model.predict(X)

# Compare with previously saved predictions
old_predictions = data["linear_prediction"].values

max_difference = np.max(
    np.abs(new_predictions - old_predictions)
)

print("Member B Handover Verification")
print("-" * 40)
print("Model loaded successfully!")
print("Prediction count:", len(new_predictions))
print(f"Maximum difference: {max_difference:.10f}")

if np.allclose(new_predictions, old_predictions):
    print("\nHANDOVER TEST PASSED!")
else:
    print("\nHANDOVER TEST FAILED!")
