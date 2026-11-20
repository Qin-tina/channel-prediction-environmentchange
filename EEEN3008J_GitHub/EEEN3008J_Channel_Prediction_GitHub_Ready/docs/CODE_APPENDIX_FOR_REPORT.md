# Appendix — Python Code and Reproducibility

**Project:** Channel Prediction That Survives an Environment Change (EEEN3008J)

The course brief permits code in a GitHub repository linked in the report **or** as appendices excluded from the page limit. This appendix is a code directory and runbook accompanying the supplied source files. The actual full scripts remain in this repository; this directory is not a substitute for attaching them if an assessor requires a self-contained appendix.

## A. Member A — Channel simulation

- Source: `ABC/01_Original_A/Member_A_Deliverables/generate_channels.py`
- Data: `ABC/01_Original_A/Member_A_Deliverables/outputs/`
- Scope: environment A/B simulation, training/validation/test CSVs.
- The Member A source and datasets are preserved **without modification**.

## B. Member B — Prediction baselines

- Source folder: `ABC/02_Original_B/Member_B_Deliverables/`
- Main scripts: `train_and_test.py`, `environment_change_test.py`, `correlation_sweep.py`, `check_predictions.py`, `save_model.py`.
- Original serialized model: `fixed_linear_model.pkl`.
- Original B predictions: `environment_B_predictions.csv`.
- The Member B source, artifacts and model are preserved **without modification**.

## C. Member C — Fallback mechanism

- Main script: `ABC/03_Updated_C/run_C_complete.py`
- Methods: last-value, fixed linear, basic residual-triggered fallback, and performance-aware residual fallback.
- Outputs: `ABC/03_Updated_C/results/`.
- The enhanced policy can preserve the fixed linear baseline on B's supplied trace; its 0% fallback rate is not proof of performance improvement over that baseline.

## D. Member D — Link adaptation and results

- Main script: `D/part_d.py`
- Outputs: `D/results/` and `D/figures/`.
- The rate-selection and packet-success abstractions are illustrative; see `D/results/method_notes.txt`.

## Reproduction

From the repository root, using a compatible Python interpreter:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The launcher executes C followed by D using existing A/B outputs; it does not retrain B or overwrite A/B source artifacts. If a version warning occurs when loading the provided scikit-learn pickle, consult B's original dependency specifications.

## Integrity and statistical notes

- The individual A/B files match the original member-supplied ZIP archives byte for byte; see `ORIGINAL_A_B_INTEGRITY.csv`.
- B's original Environment B prediction file is one trace, not 12 independent trials. A-produced traces use different simulation settings and must not be conflated with the original B test as one matched A/B experiment.
- Code availability must only contain a **real** GitHub URL after the repository has been published. A placeholder link is not a published URL.

## Report statement — if repository is published

> The implementation scripts and simulation data for Parts A–D, together with reproducibility instructions, are available at [paste the real GitHub repository URL here].

## Report statement — if code is submitted as appendices instead

> The Python implementation scripts for Parts A–D, associated simulation data, and reproducibility instructions are provided in the supplementary Code Appendix accompanying this report.
