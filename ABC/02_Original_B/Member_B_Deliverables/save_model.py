import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load Environment A training data
train_snr = pd.read_csv("train_snr.csv")["snr_db"].values

# Prepare training samples
X_train = train_snr[:-1].reshape(-1, 1)
y_train = train_snr[1:]

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fixed_linear_model.pkl")

print("Model saved successfully!")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print("Saved file: fixed_linear_model.pkl")