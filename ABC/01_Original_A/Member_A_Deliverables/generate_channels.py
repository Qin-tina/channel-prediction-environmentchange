
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================
# 1. Simulation settings
# ==============================================

SEED = 2026
N = 1500
DT_MS = 5
LAGS = 5

MEAN_SNR_DB = 12.0
STD_SNR_DB = 4.0
MEAS_NOISE_DB = 0.6

RHO_A = 0.96
RHO_B = 0.65
MAX_ACF_LAG = 30

out = Path("outputs")
out.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# ==============================================
# 2. AR(1) channel-quality generator
# ==============================================

def generate_trace(rho):

    z = np.empty(N)
    z[0] = rng.normal()

    eps = rng.normal(size=N - 1)

    for t in range(1, N):
        z[t] = (
            rho * z[t - 1]
            + np.sqrt(1 - rho**2) * eps[t - 1]
        )

    true_snr = MEAN_SNR_DB + STD_SNR_DB * z

    measured_snr = (
        true_snr
        + rng.normal(0, MEAS_NOISE_DB, N)
    )

    return true_snr, measured_snr

# ==============================================
# 3. Empirical autocorrelation function
# ==============================================

def acf(trace, max_lag):

    x = np.asarray(trace) - np.mean(trace)
    variance = np.mean(x**2)

    return np.array([
        np.mean(x[:len(x)-lag] * x[lag:]) / variance
        if lag else 1.0
        for lag in range(max_lag + 1)
    ])

# ==============================================
# 4. Construct independent datasets
# ==============================================

all_traces = {}
summary = []

def build_split(name, rho, n_sequences, environment):

    rows = []
    traces = []

    for seq in range(n_sequences):

        true_snr, measured_snr = generate_trace(rho)
        traces.append((true_snr, measured_snr))

        for t in range(LAGS - 1, N - 1):

            record = {
                "environment": environment,
                "sequence_id": f"{name}_{seq:03d}",
                "time_index": t
            }

            for lag in range(LAGS):
                record[f"snr_lag{lag}_db"] = (
                    measured_snr[t - lag]
                )

            record["target_true_snr_next_db"] = (
                true_snr[t + 1]
            )

            rows.append(record)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{name}.csv", index=False)

    all_traces[name] = traces

    sample_corr = np.mean([
        np.corrcoef(t[:-1], t[1:])[0, 1]
        for t, _ in traces
    ])

    summary.append({
        "dataset": name,
        "environment": environment,
        "rho_configured": rho,
        "n_sequences": n_sequences,
        "prediction_rows": len(df),
        "mean_lag1_correlation": sample_corr
    })

# ==============================================
# 5. Generate A/B datasets
# ==============================================

build_split("train_A", RHO_A, 24, "A")
build_split("val_A", RHO_A, 8, "A")
build_split("test_A", RHO_A, 12, "A")
build_split("test_B", RHO_B, 12, "B")

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    out / "dataset_summary.csv",
    index=False
)

print("DATASET VERIFICATION")
print(summary_df.to_string(index=False))

# ==============================================
# 6. Figure 1: Channel-quality sequences
# ==============================================

fig, axes = plt.subplots(
    2, 1,
    figsize=(10, 6.5),
    sharex=True,
    sharey=True
)

for ax, name, rho in zip(
    axes,
    ["test_A", "test_B"],
    [RHO_A, RHO_B]
):

    true_snr, measured_snr = all_traces[name][0]

    length = 160
    time_ms = np.arange(length) * DT_MS

    ax.plot(
        time_ms,
        true_snr[:length],
        linewidth=1.8,
        label="Underlying SNR"
    )

    ax.plot(
        time_ms,
        measured_snr[:length],
        linewidth=0.95,
        alpha=0.65,
        label="Measured SNR"
    )

    env = "A" if name == "test_A" else "B"

    ax.set_title(
        f"Environment {env} "
        f"(correlation coefficient = {rho})"
    )

    ax.set_ylabel("SNR (dB)")
    ax.set_ylim(-5, 28)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Time (ms)")

fig.tight_layout()

fig.savefig(
    out / "figure1_channel_traces.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# ==============================================
# 7. Figure 2: Temporal autocorrelation
# ==============================================

fig, ax = plt.subplots(figsize=(9, 5.5))

lags = np.arange(MAX_ACF_LAG + 1)

for name, rho, marker in [
    ("test_A", RHO_A, "o"),
    ("test_B", RHO_B, "s")
]:

    empirical = np.mean(
        [
            acf(true, MAX_ACF_LAG)
            for true, _ in all_traces[name]
        ],
        axis=0
    )

    env = "A" if name == "test_A" else "B"

    line, = ax.plot(
        lags,
        empirical,
        marker=marker,
        markevery=3,
        linewidth=1.9,
        label=f"Environment {env}: empirical"
    )

    ax.plot(
        lags,
        rho**lags,
        "--",
        linewidth=1.7,
        color=line.get_color(),
        label=f"Environment {env}: theoretical"
    )

ax.set_xlabel("Time lag (samples)")
ax.set_ylabel("Autocorrelation coefficient")

ax.set_title(
    "Temporal autocorrelation: "
    "empirical vs. AR(1) theory"
)

ax.set_ylim(-0.1, 1.05)
ax.grid(True, alpha=0.25)
ax.legend()

fig.tight_layout()

fig.savefig(
    out / "figure2_autocorrelation.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("All datasets and figures generated.")

from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

def draw_box(x, y, w, h, text):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.5,
        edgecolor="#2563eb",
        facecolor="#eff6ff"
    )
    ax.add_patch(rect)
    ax.text(
        x + w/2, y + h/2,
        text,
        ha="center", va="center",
        fontsize=11
    )

def arrow(x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            lw=1.8
        )
    )

draw_box(3, 8.5, 4, 0.9, "AR(1) Channel Generator")

draw_box(0.5, 6.3, 4, 1.0,
         "Environment A\nrho = 0.96")

draw_box(5.5, 6.3, 4, 1.0,
         "Environment B\nrho = 0.65")

draw_box(2.5, 3.8, 5, 1.0,
         "Prediction Methods\n"
         "Last-value / Linear / Fallback")

draw_box(2.5, 1.3, 5, 1.0,
         "Performance Evaluation\n"
         "RMSE / Goodput / Fallback Frequency")

arrow(5, 8.5, 2.5, 7.3)
arrow(5, 8.5, 7.5, 7.3)

arrow(2.5, 6.3, 4, 4.8)
arrow(7.5, 6.3, 6, 4.8)

arrow(5, 3.8, 5, 2.3)

fig.savefig(
    out / "figure3_framework.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
