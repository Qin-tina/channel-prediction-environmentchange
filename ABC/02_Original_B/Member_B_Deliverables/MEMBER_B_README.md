
# Member B - Channel Prediction Models

## Project
Channel Prediction That Survives an Environment Change

## Responsibilities
- Implement Last-value Prediction
- Train a Linear Regression Predictor
- Evaluate prediction RMSE
- Test prediction under environment changes
- Export the trained prediction model

## Implemented Methods

### 1. Last-value Prediction
Predicts the next SNR using the most recent SNR measurement.

### 2. Linear Regression
A one-step-ahead linear predictor trained on Environment A
(rho = 0.95).

Input: Previous SNR measurement (dB)

Output: Predicted next SNR (dB)

## Saved Model
File: fixed_linear_model.pkl

The model can be loaded using:

```python
import joblib

model = joblib.load("fixed_linear_model.pkl")
prediction = model.predict([[10.0]])

print(prediction)
```

## Experimental Results

| Channel correlation | Last-value RMSE | Linear RMSE |
|---|---|---|
| 0.95 | 0.6238 | 0.6192 |
| 0.80 | 1.2459 | 1.2073 |
| 0.60 | 1.7602 | 1.6986 |
| 0.40 | 2.1571 | 2.0786 |

RMSE values are measured in dB.

Each condition uses 10 independently generated test sequences.

## Files for Group Members

- fixed_linear_model.pkl: Trained Linear Regression model
- environment_B_predictions.csv: Predictions under changed conditions
- correlation_sweep_results.csv: RMSE and 95% confidence intervals
- correlation_sweep_95ci.png: Prediction performance figure
- load_model_test.py: Model loading verification

## Important Notes

- The current channel data is synthetic.
- The predictor is trained only on Environment A.
- The current model uses one previous SNR measurement.
- No model retraining is performed during testing.
- The reported results are preliminary.
- Final experiments should use the group's agreed channel model
  and independent training, validation, and test sequences.
- Packet goodput evaluation and fallback evaluation are handled
  by other group members.
  
## Instructions for Member C and Member D

### For Member C: Fallback Mechanism

Use the following files:

- fixed_linear_model.pkl
- environment_B_predictions.csv

The CSV contains:

- time_step: Prediction time step
- actual_snr: Actual channel SNR in dB
- last_value_prediction: Last-value prediction
- linear_prediction: Linear Regression prediction

Member C can use these values to calculate
prediction residuals and develop a fallback
mechanism.

Important: A residual can only be calculated
after the corresponding actual SNR is observed.

### For Member D: Goodput Evaluation

Use environment_B_predictions.csv.

Both prediction methods must be evaluated
using the same rate-selection table and
the same actual channel conditions.

The predicted SNR determines the selected
transmission rate.

The actual SNR determines whether the
transmission succeeds under the chosen
link-performance model.

### Model Information

Model type: Linear Regression

Training environment:
Channel correlation coefficient = 0.95

Input:
One previous SNR measurement (dB)

Output:
Predicted next SNR (dB)

Model parameters:
Coefficient = 0.9234
Intercept = 0.7325

### Reproducibility

Required Python packages:

numpy
pandas
scikit-learn
scipy
matplotlib
joblib

The current results use synthetic channel data.

The saved model was successfully reloaded,
and its predictions matched the original
outputs.

These are preliminary simulation results.
Final experiments should follow the group's
agreed data generation and evaluation protocol.

