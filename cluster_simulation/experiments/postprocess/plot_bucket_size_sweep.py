"""Plots the token bucket size sweep: how SLO attainment and admission respond as
the bucket size moves from a quarter to twice the size TOKEN_BUCKET derives.

Writes, into [src]/figures/:
    bucket_size_sweep.png   SLO attainment and rejection rate against bucket size,
                            one column per workflow, const and burst as lines
    <family>_<workload>_tail.png
                            tail CDF of normalized response time per bucket size,
                            shaded light to dark from 0.25x to 2x
and [src]/summary.csv with one row per run.

Reads the results of experiments/run_bucket_size_sweep.py.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_bucket_size_sweep.py \
        --src experiments/results/bucket_size_sweep
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FAMILIES = [("ppl2", "WF1 (pipeline 2)"), ("wf6", "WF6 (textvision)"),
            ("wf12", "WF12 (defect inspection)"), ("wf13", "WF13 (cow teat health)")]
# runs missing for a scale (0.125x exists only for WF6) are skipped
SCALES = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# categorical slots 1 and 2, as in plot_slo_sweep.py
SERIES = [("const", "constant", "#2a78d6"), ("burst", "bursty", "#eb6834")]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"


def summarize(src: str) -> pd.DataFrame:
    """Returns one row per run: its bucket size and how its jobs fared."""
    rows = []
    for family, _ in FAMILIES:
        for workload, _, _ in SERIES:
            for scale in SCALES:
                run = os.path.join(src, f"{family}_{workload}_{scale:g}x", "sim_logs")
                if not os.path.isdir(run):
                    continue

                jobs = pd.read_csv(os.path.join(run, "job_log.csv"))
                done = jobs["was_completed"].astype(bool)
                met = done & (jobs["response_time"] <= jobs["deadline"] - jobs["create_time"])
                adm = pd.read_csv(os.path.join(run, "admission_control.csv")).iloc[0]

                rows.append({"family": family, "workload": workload, "scale": scale,
                             "burst_size_jobs": adm["burst_size_jobs"],
                             "jobs": len(jobs),
                             "rejected": adm["num_rejected"] / len(jobs),
                             "completed": done.mean(), "slo_met": met.mean(),
                             "p99_ms": jobs.loc[done, "response_time"].quantile(.99)})

    return pd.DataFrame(rows)


def _style(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")


def plot_summary(summary: pd.DataFrame, out_path: str):
    """SLO attainment (top) and share of jobs rejected at admission (bottom)
    against bucket size, one column per workflow.
    """
    # only the workflows this sweep directory actually ran
    families = [(f, n) for f, n in FAMILIES if (summary["family"] == f).any()]

    fig, axes = plt.subplots(2, len(families), figsize=(4 * len(families), 7.5),
                             sharex=True, sharey="row", squeeze=False)

    for col, (family, name) in enumerate(families):
        rows = summary[summary["family"] == family]
        derived = rows.loc[rows["scale"] == 1.0, "burst_size_jobs"]
        title = f"{name}\nderived bucket: {derived.iloc[0]:.1f} jobs" if len(derived) else name

        for row, (metric, ylabel) in enumerate(
                [("slo_met", "Jobs meeting the SLO (%)"),
                 ("rejected", "Jobs rejected at admission (%)")]):
            ax = axes[row][col]
            _style(ax, title if row == 0 else "")

            # the size TOKEN_BUCKET derives on its own
            ax.axvline(1.0, color=INK_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))

            for workload, label, color in SERIES:
                pts = rows[rows["workload"] == workload].sort_values("scale")
                ax.plot(pts["scale"], 100 * pts[metric], label=label, color=color,
                        linewidth=2, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=2)

            ax.set_xscale("log", base=2)
            ax.set_xticks(SCALES)
            # the log axis packs 1.25x-2x together, so only every other one is labeled
            ax.set_xticklabels([f"{s:g}×" if s not in (1.25, 1.75) else "" for s in SCALES])
            ax.minorticks_off()
            if col == 0:
                ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
            if row == 1:
                ax.set_xlabel("Bucket size (multiple of derived)", color=INK_SECONDARY,
                              fontsize=10)

    axes[1][0].set_ylim(0, None)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
                      loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_tails(src: str, fig_dir: str):
    """One normalized tail CDF per workflow and workload, a series per bucket size
    shaded light (0.25x) to dark (2x)."""
    plot_tail = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_tail.py")
    for family, _ in FAMILIES:
        for workload, _, _ in SERIES:
            runs = [(s, os.path.join(src, f"{family}_{workload}_{s:g}x", "sim_logs"))
                    for s in SCALES]
            runs = [(s, r) for s, r in runs if os.path.isdir(r)]
            if not runs:
                continue
            subprocess.run(
                [sys.executable, plot_tail, "--pdf", "--normalize", "--xmax", "10",
                 "--srcs"] + [r for _, r in runs]
                + ["--labels"] + [f"bucket {s:g}×" for s, _ in runs]
                + ["--groups"] + ["bucket size"] * len(runs)
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
    plot_summary(summary, os.path.join(fig_dir, "bucket_size_sweep.png"))
    plot_tails(args.src, fig_dir)
    print(f"wrote {args.src}/summary.csv and {fig_dir}/")
