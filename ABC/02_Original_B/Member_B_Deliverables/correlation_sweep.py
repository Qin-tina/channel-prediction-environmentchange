
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t

# ==========================================
# Step 1: Load Environment A training data
# ==========================================

train_snr = pd.read_csv("train_snr.csv")["snr_db"].values

X_train = train_snr[:-1].reshape(-1, 1)
y_train = train_snr[1:]

# Train Linear Regression only on Environment A
model = LinearRegression()
model.fit(X_train, y_train)

print("Linear Regression trained successfully!")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")


# ==========================================
# Step 2: Set experiment parameters
# ==========================================

rho_values = [0.95, 0.80, 0.60, 0.40]

n_runs = 10
n_samples = 1000

last_rmse_results = []
linear_rmse_results = []

last_ci_results = []
linear_ci_results = []


# ==========================================
# Step 3: Run experiments
# ==========================================

for rho in rho_values:

    last_run_rmse = []
    linear_run_rmse = []

    for run in range(n_runs):

        # Independent sequence for each run
        rng = np.random.default_rng(1000 + run)

        snr = np.zeros(n_samples)
        snr[0] = 10.0

        # Generate correlated channel-quality data
        for i in range(1, n_samples):
            snr[i] = (
                10
                + rho * (snr[i - 1] - 10)
                + np.sqrt(1 - rho**2) * rng.normal(0, 2)
            )

        # Prepare one-step-ahead predictions
        X_test = snr[:-1].reshape(-1, 1)
        y_test = snr[1:]

        # Last-value prediction
        last_pred = X_test.flatten()

        # Fixed Linear Regression prediction
        linear_pred = model.predict(X_test)

        # Calculate RMSE for this run
        last_rmse = np.sqrt(
            np.mean((y_test - last_pred) ** 2)
        )

        linear_rmse = np.sqrt(
            np.mean((y_test - linear_pred) ** 2)
        )

        last_run_rmse.append(last_rmse)
        linear_run_rmse.append(linear_rmse)

    # ==========================================
    # Step 4: Average RMSE for each environment
    # ==========================================

    last_mean = np.mean(last_run_rmse)
    linear_mean = np.mean(linear_run_rmse)

    last_rmse_results.append(last_mean)
    linear_rmse_results.append(linear_mean)

    # ==========================================
    # Step 5: Calculate 95% confidence intervals
    # ==========================================

    t_critical = t.ppf(0.975, df=n_runs - 1)

    last_ci = t_critical * (
        np.std(last_run_rmse, ddof=1) / np.sqrt(n_runs)
    )

    linear_ci = t_critical * (
        np.std(linear_run_rmse, ddof=1) / np.sqrt(n_runs)
    )

    last_ci_results.append(last_ci)
    linear_ci_results.append(linear_ci)


# ==========================================
# Step 6: Display experiment results
# ==========================================

print("\nCorrelation Sweep Results with 95% CI")
print("-" * 70)

for i, rho in enumerate(rho_values):

    print(
        f"rho={rho:.2f} | "
        f"Last-value: {last_rmse_results[i]:.4f} "
        f"+/- {last_ci_results[i]:.4f} dB | "
        f"Linear: {linear_rmse_results[i]:.4f} "
        f"+/- {linear_ci_results[i]:.4f} dB"
    )


# ==========================================
# Step 7: Save experiment results
# ==========================================

results = pd.DataFrame({
    "rho": rho_values,
    "last_value_rmse": last_rmse_results,
    "linear_rmse": linear_rmse_results,
    "last_value_ci95": last_ci_results,
    "linear_ci95": linear_ci_results
})

results.to_csv(
    "correlation_sweep_results.csv",
    index=False
)

print("\nResults saved to correlation_sweep_results.csv")


# ==========================================
# Step 8: Plot RMSE with 95% confidence intervals
# ==========================================

plt.figure(figsize=(9, 5))

plt.errorbar(
    rho_values,
    last_rmse_results,
    yerr=last_ci_results,
    marker="o",
    capsize=5,
    linewidth=2,
    label="Last-value Prediction"
)

plt.errorbar(
    rho_values,
    linear_rmse_results,
    yerr=linear_ci_results,
    marker="s",
    capsize=5,
    linewidth=2,
    label="Linear Regression"
)

plt.xlabel("Channel Correlation Coefficient (rho)")
plt.ylabel("RMSE (dB)")
plt.title("Prediction RMSE under Environment Changes (95% CI)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "correlation_sweep_95ci.png",
    dpi=300,
    bbox_inches="tight"
)

print("Figure saved to correlation_sweep_95ci.png")

plt.show()
