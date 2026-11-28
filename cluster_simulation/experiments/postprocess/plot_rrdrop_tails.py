"""Plots the core tail CDFs with the RR-drop baseline rerun by run_rrdrop_sweep.py,
at the share of jobs the best policy (tuned token buckets included) leaves unserved.

The figures are plot_core_tails.py's, with the RR-drop line taken from the rerun and
labelled with its drop rate, for every family and workload run_rrdrop_sweep.py
covers, AliTrace included. The default token bucket is left out: the optimal and
adaptive token buckets stand for token bucket admission.

Writes, into [src]/figures/:
    <name>_<workload>_tail.png      normalized tail CDF
    <name>_<workload>_tail_ms.png   tail CDF in ms (families in MS_PLOTS)

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_rrdrop_tails.py \\
        --src experiments/results/rrdrop_min_unserved
"""

import argparse
import json
import os
import subprocess
import sys

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_core_tails as pct
from run_rrdrop_sweep import ADAPTIVE_DIRS, CORE_DIRS, NAMES

# plot_tail options for every figure: a dashed line at the SLO, and the legend
# under the plot, right-aligned, in two columns
OPTIONS = ["--slo-line", "--legend-fontsize", "13", "--legend-loc", "below right",
           "--legend-ncol", "2", "--minor-labels"]

# families whose SLO marker is drawn bold: their SLO falls among the curves' steep
# drops, where the default marker is lost
BOLD_SLO = {"wf13", "wf15"}

# right edge of the ms figures, in multiples of the SLO
MS_XMAX_SLOS = 2

# core policies not drawn: the default token bucket, which the optimal and adaptive
# token buckets tune
SKIPPED = {"token"}

# the RR-drop baseline rejects jobs at admission, so these figures name it, and its
# drop rate, as admission control
RRDROP_LABEL = "RR admission control"


def rrdrop_rate(src: str, family: str, workload: str) -> float:
    """The share of jobs the RR-drop rerun of [family] under [workload] rejects."""
    with open(os.path.join(src, f"{family}_{workload}_rrdrop", "overrides.json")) as f:
        return json.load(f)["overrides"]["ADMISSION_DROP_RATE"]


def policy_runs(src: str, family: str, workload: str) -> tuple[list[str], list[str]]:
    """Returns the sim_logs directory and legend label of every policy the figures
    draw for [family] under [workload], in legend order."""
    core_dir = CORE_DIRS[(family, workload)]
    rrdrop = os.path.join(src, f"{family}_{workload}_rrdrop")
    drop_rate = rrdrop_rate(src, family, workload)

    srcs, labels = [], []
    for policy, label in pct.POLICIES:
        if policy in SKIPPED:
            continue
        if policy == "adaptive":
            run = os.path.join(pct.RESULTS, ADAPTIVE_DIRS[workload],
                               f"{family}_{workload}_adaptive")
            if not os.path.isdir(run):
                continue
            label = pct.adaptive_label(label, run)
        elif policy == "optimal":
            run = pct.optimal_run(family, workload)
        elif policy == "rrdrop":
            run, label = rrdrop, f"{RRDROP_LABEL} ({100 * drop_rate:.1f}% rejected)"
        else:
            run = os.path.join(pct.RESULTS, core_dir, f"{family}_{workload}_{policy}")

        logs = os.path.join(run, "sim_logs")
        assert(os.path.isdir(logs)), f"Missing results for {label}: {logs}"
        if policy in ("token", "optimal"):
            label = pct.bucket_label(label, run)
        srcs.append(logs)
        labels.append(label)
    return srcs, labels


def plot(src: str, family: str, workload: str):
    name = NAMES[family]
    drop_rate = rrdrop_rate(src, family, workload)
    srcs, labels = policy_runs(src, family, workload)

    fig_dir = os.path.join(src, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_tail = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_tail.py")
    variants = [(pct.NORMALIZED[family], f"{name}_{workload}_tail.png")]
    if family in pct.MS_PLOTS:
        variants.append((["--xmax-slos", str(MS_XMAX_SLOS)], f"{name}_{workload}_tail_ms.png"))

    for options, out in variants:
        subprocess.run([sys.executable, plot_tail, "--pdf"] + OPTIONS + options
                       + (["--bold-slo"] if family in BOLD_SLO else [])
                       + ["--srcs"] + srcs + ["--labels"] + labels
                       + ["--out", os.path.join(fig_dir, out)], check=True)
    print(f"{name} {workload}: RR-drop at {100 * drop_rate:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=str, required=True,
                        help="run_rrdrop_sweep.py results directory")
    args = parser.parse_args()

    for family, workload in CORE_DIRS:
        plot(args.src, family, workload)
