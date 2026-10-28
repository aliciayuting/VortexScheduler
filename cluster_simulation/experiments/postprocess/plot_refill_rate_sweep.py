"""Plots the token bucket refill rate sweep: how SLO attainment and admission
respond as the refill rate moves from 0.8 to 1.1 times the estimated capacity,
with the bucket size held fixed.

Writes, into [src]/figures/:
    refill_rate_sweep.png   SLO attainment and rejection rate against refill rate,
                            one column per workflow, const and burst as lines
    <family>_<workload>_tail.png
                            tail CDF of normalized response time per refill rate,
                            shaded light to dark from the lowest rate to the highest
and [src]/summary.csv with one row per run.

Reads the results of experiments/run_refill_rate_sweep.py.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_refill_rate_sweep.py \
        --src experiments/results/refill_rate_sweep
"""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt

from plot_bucket_size_sweep import (FAMILIES, SERIES, TAIL_XMAX, INK_SECONDARY,
                                    SURFACE, _style)

RATES = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]


def _run_dir(src: str, family: str, workload: str, rate: float) -> str:
    return os.path.join(src, f"{family}_{workload}_rate{rate:g}")


def summarize(src: str) -> pd.DataFrame:
    """Returns one row per run: its refill rate and how its jobs fared."""
    rows = []
    for family, _ in FAMILIES:
        for workload, _, _ in SERIES:
            for rate in RATES:
                run = _run_dir(src, family, workload, rate)
                if not os.path.isdir(os.path.join(run, "sim_logs")):
                    continue

                jobs = pd.read_csv(os.path.join(run, "sim_logs", "job_log.csv"))
                done = jobs["was_completed"].astype(bool)
                met = done & (jobs["response_time"] <= jobs["deadline"] - jobs["create_time"])
                adm = pd.read_csv(os.path.join(run, "sim_logs", "admission_control.csv")).iloc[0]
                with open(os.path.join(run, "overrides.json")) as f:
                    burst_scale = json.load(f)["overrides"]["ADMISSION_BURST_SCALE"]

                rows.append({"family": family, "workload": workload, "rate": rate,
                             # the run whose bucket was left unscaled is the one at
                             # the configured default rate
                             "is_default": abs(burst_scale - 1) < 1e-9,
                             "admit_rate_qps": adm["admit_rate_qps"],
                             "burst_size_jobs": adm["burst_size_jobs"],
                             "jobs": len(jobs),
                             "rejected": adm["num_rejected"] / len(jobs),
                             "completed": done.mean(), "slo_met": met.mean(),
                             "p99_ms": jobs.loc[done, "response_time"].quantile(.99)})

    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_path: str):
    """SLO attainment (top) and share of jobs rejected at admission (bottom)
    against refill rate, one column per workflow.
    """
    # only the workflows and rates this sweep directory actually ran
    families = [(f, n) for f, n in FAMILIES if (summary["family"] == f).any()]
    rates = [r for r in RATES if (summary["rate"] == r).any()]
    defaults = summary.loc[summary["is_default"], "rate"].unique()

    fig, axes = plt.subplots(2, len(families), figsize=(4 * len(families), 7.5),
                             sharex=True, sharey="row", squeeze=False)

    for col, (family, name) in enumerate(families):
        rows = summary[summary["family"] == family]
        bucket = rows["burst_size_jobs"].iloc[0]
        _style(axes[0][col], f"{name}\nbucket: {bucket:.1f} jobs")
        _style(axes[1][col], "")

        for row, (metric, ylabel) in enumerate(
                [("slo_met", "Jobs meeting the SLO (%)"),
                 ("rejected", "Jobs rejected at admission (%)")]):
            ax = axes[row][col]

            # the rate TOKEN_BUCKET uses by default, and the capacity estimate itself
            for rate in defaults:
                ax.axvline(rate, color=INK_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
            ax.axvline(1.0, color=INK_SECONDARY, linewidth=1, linestyle=(0, (1, 2)))

            for workload, label, color in SERIES:
                pts = rows[rows["workload"] == workload].sort_values("rate")
                ax.plot(pts["rate"], 100 * pts[metric], label=label, color=color,
                        linewidth=2, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=2)

            ax.set_xticks(rates)
            # every other rate is labeled, so the ticks do not crowd
            ax.set_xticklabels([f"{r:g}" if i % 2 == 0 else "" for i, r in enumerate(rates)])
            if col == 0:
                ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
            if row == 1:
                ax.set_xlabel("Refill rate (fraction of estimated capacity)",
                              color=INK_SECONDARY, fontsize=10)

    # name the two reference lines once, in the first panel
    ax = axes[0][0]
    for rate in defaults:
        ax.annotate("default", (rate, 1), xycoords=("data", "axes fraction"),
                    xytext=(-3, -3), textcoords="offset points", ha="right", va="top",
                    color=INK_SECONDARY, fontsize=8)
    ax.annotate("capacity\nestimate", (1.0, 1), xycoords=("data", "axes fraction"),
                xytext=(3, -3), textcoords="offset points", ha="left", va="top",
                color=INK_SECONDARY, fontsize=8)

    axes[1][0].set_ylim(0, None)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_tails(src: str, fig_dir: str):
    """One normalized tail CDF per workflow and workload, a series per refill rate
    shaded light (lowest) to dark (highest)."""
    plot_tail = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_tail.py")
    for family, _ in FAMILIES:
        for workload, _, _ in SERIES:
            runs = [(r, os.path.join(_run_dir(src, family, workload, r), "sim_logs"))
                    for r in RATES]
            runs = [(r, d) for r, d in runs if os.path.isdir(d)]
            if not runs:
                continue
            subprocess.run(
                [sys.executable, plot_tail, "--pdf", "--normalize",
                 "--xmax", str(TAIL_XMAX.get(family, 10)),
                 "--srcs"] + [d for _, d in runs]
                + ["--labels"] + [f"refill {r:g}" for r, _ in runs]
                + ["--groups"] + ["refill rate"] * len(runs)
                + ["--out", os.path.join(fig_dir, f"{family}_{workload}_tail.png")],
                check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, required=True, help="Sweep results directory")
    args = parser.parse_args()

    fig_dir = os.path.join(args.src, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    summary = summarize(args.src)
    summary.to_csv(os.path.join(args.src, "summary.csv"), index=False)
    plot_summary(summary, os.path.join(fig_dir, "refill_rate_sweep.png"))
    plot_tails(args.src, fig_dir)
    print(f"wrote {args.src}/summary.csv and {fig_dir}/")
