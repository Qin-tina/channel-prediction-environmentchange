[README.md](https://github.com/user-attachments/files/35139937/README.md)
# Channel Prediction under Environment Change

EEEN3008J Wireless Communications — group simulation project.

This repository contains channel traces, prediction baselines, a residual-triggered fallback method, and a packet-goodput evaluation. Source files for Parts A and B are kept in their original directories; the experiment stages use the saved outputs rather than retraining the fixed linear model.

## Repository contents

| Part | Location | Description |
| --- | --- | --- |
| A | `ABC/01_Original_A/Member_A_Deliverables/` | Channel-trace generation and Environment A/B data |
| B | `ABC/02_Original_B/Member_B_Deliverables/` | Last-value and linear prediction, trained model and prediction tables |
| C | `ABC/03_Updated_C/` | Basic and performance-aware residual-triggered fallback |
| D | `D/` | Rate selection, packet-goodput calculations and result figures |

`run_all.py` runs Parts C and D in sequence using the existing A/B files. It does not retrain the B model. The original scripts for A and B are also available for individual inspection.

## Requirements

A working Python environment with packages listed in `requirements.txt` is required. The saved B model was produced with an earlier version of scikit-learn; loading it with a different version may raise a compatibility warning. Refer to B's `requirements.txt` when reproducing its original training workflow.

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Generated files can be found in `ABC/03_Updated_C/results/` and `D/results/`, with D's figures in `D/figures/`.

## Experimental interpretation

The original B Environment B predictions and A's channel traces are based on different simulation settings. Results from the two sources are reported separately. The single B test trace does not support a multi-run confidence interval. On that trace, the enhanced fallback retains the fixed linear predictor instead of switching to a less accurate last-value predictor. This should not be interpreted as an improvement beyond the fixed linear baseline.

The packet-goodput model uses an illustrative rate-selection table and simplified packet-success criterion. Details are in `D/results/method_notes.txt`.

## Report reference

**Source code and data:** https://github.com/Qin-tina/channel-prediction-environmentchange

Publication of this repository should be agreed upon by the contributors and permitted by the course assessment rules.
