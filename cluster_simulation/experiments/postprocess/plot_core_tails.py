"""Plots the core sweep's tail CDFs, one figure per workflow and workload, with
the best static token bucket alongside the core policies.

The best static token bucket ("token-bucket (optimal)") is the run with the
highest SLO attainment for that workflow and workload across the bucket size
sweeps in OPTIMAL_SWEEPS: bucket sizes swept at the refill rate the refill rate
sweep found best (bucket_size_sweep_optrate), at alternative rates
(bucket_size_sweep_altrate) and at the capacity estimate
(bucket_size_sweep_rate1). It is picked from those sweeps' summary.csv files, so
it follows them if they are rerun.

Writes, into each core results directory's figures/:
    <name>_<workload>_tail.png      normalized tail CDF (all families)
    <name>_<workload>_tail_ms.png   tail CDF in ms (families in MS_PLOTS)

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_core_tails.py
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

# bucket size sweeps the optimal token bucket is picked from, all run by
# run_bucket_size_sweep.py on the core sweep's arrival traces: at the refill rate
# the refill rate sweep found best, at the alternative rates that came close, and
# at the capacity estimate itself. The AliTrace and WF15 staggered line workloads'
# sweeps hold only their own runs, at the refill rates their refill rate sweeps
# favored. Earlier sweeps win ties
OPTIMAL_SWEEPS = [os.path.join(RESULTS, sweep) for sweep in
                  ("bucket_size_sweep_optrate", "bucket_size_sweep_altrate",
                   "bucket_size_sweep_rate1", "bucket_size_sweep_alitrace",
                   "bucket_size_sweep_wf15_staggered")]

# adaptive token bucket runs (run_adaptive_bucket_sweep.py with a 200 job count
# window and --grow-only-if-rejecting). Families it was not run for get no line
ADAPTIVE_SWEEP = os.path.join(RESULTS, "adaptive_bucket_rejgrow")

# core results directory, family as run names give it, and name used in the figures
CORE_RUNS = [("wf1_wf13_core", "ppl2", "wf1"),
             ("wf6_core", "wf6", "wf6"),
             ("wf1_wf13_core", "wf13", "wf13"),
             ("wf14_core", "wf14", "wf14"),
             ("wf15_core", "wf15", "wf15")]

# core policies plotted, in legend order. The optimal token bucket goes in right
# after the token bucket it tunes
POLICIES = [("nodrop", "no-drop"), ("early", "EARLY"), ("nexus", "Nexus"),
            ("lookahead", "LOOKAHEAD"), ("token", "token-bucket"),
            ("optimal", "token-bucket (optimal)"),
            ("adaptive", "token-bucket (adaptive)"), ("rrdrop", "RR-drop")]

# plot_tail options for the normalized figure of each family
NORMALIZED = {"ppl2": ["--normalize", "--xmax", "10"],
              "wf6": ["--normalize", "--xmax", "10"],
              "wf13": ["--normalize", "--xmax", "10"],
              # WF14's inputs arrive over time; measure what remains after that wait
              "wf14": ["--normalize", "--minus-delay", "--xmax", "12"],
              "wf15": ["--normalize", "--xmax", "3.5"]}

# families also plotted in absolute ms
MS_PLOTS = {"wf15"}


def optimal_run(family: str, workload: str) -> str:
    """Returns the run of OPTIMAL_SWEEPS with the highest SLO attainment for
    [family] and [workload]."""
    best = optimal_bucket(family, workload)
    return os.path.join(best["sweep"], f"{family}_{workload}_{best['scale']:g}x_"
                                       f"rate{best['refill_rate']:g}")


def optimal_bucket(family: str, workload: str) -> pd.Series:
    """Returns the summary.csv row, across OPTIMAL_SWEEPS, of the static token
    bucket with the highest SLO attainment for [family] and [workload]: its
    refill rate (as a fraction of capacity), bucket scale and [sweep] directory.
    run_adaptive_bucket_sweep.py takes its refill rates from here too."""
    summaries = []
    for sweep in OPTIMAL_SWEEPS:
        summary = pd.read_csv(os.path.join(sweep, "summary.csv"))
        summary["sweep"] = sweep
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)

    rows = summary[(summary["family"] == family) & (summary["workload"] == workload)]
    assert(len(rows)), f"No {family} {workload} runs in any of {OPTIMAL_SWEEPS}"

    # idxmax takes the first of equals, i.e. the earliest sweep
    return rows.loc[rows["slo_met"].idxmax()]


def bucket_label(label: str, run: str) -> str:
    """Appends to [label] the refill rate (as a fraction of the estimated capacity)
    and bucket size (jobs) the token bucket run [run] actually used."""
    adm = pd.read_csv(os.path.join(run, "sim_logs", "admission_control.csv")).iloc[0]
    rate = adm["admit_rate_qps"] / adm["estimated_capacity_qps"]
    return f"{label} (rate {rate:.2g}×cap, bucket {adm['burst_size_jobs']:.1f})"


def adaptive_label(label: str, run: str) -> str:
    """Appends to [label] the refill rate (as a fraction of the estimated capacity)
    and the range of bucket sizes (jobs) the adaptive token bucket run [run]
    held over the run, from its starting size through every resize."""
    logs = os.path.join(run, "sim_logs")
    adm = pd.read_csv(os.path.join(logs, "admission_control.csv")).iloc[0]
    resizes = pd.read_csv(os.path.join(logs, "adaptive_bucket_log.csv"))
    rate = adm["admit_rate_qps"] / adm["estimated_capacity_qps"]
    sizes = [adm["default_burst_size_jobs"]] + list(resizes["depth_after"])
    return f"{label} (rate {rate:.2g}×cap, bucket {min(sizes):.1f}–{max(sizes):.1f})"


def plot(core_dir: str, family: str, name: str, workload: str):
    best = optimal_run(family, workload)
    print(f"{name} {workload}: optimal token bucket = {os.path.basename(best)}")

    srcs, labels = [], []
    for policy, label in POLICIES:
        if policy == "adaptive":
            run = os.path.join(ADAPTIVE_SWEEP, f"{family}_{workload}_adaptive")
            if not os.path.isdir(run):
                continue
            label = adaptive_label(label, run)
        elif policy == "optimal":
            run = best
        else:
            run = os.path.join(RESULTS, core_dir, f"{family}_{workload}_{policy}")

        logs = os.path.join(run, "sim_logs")
        assert(os.path.isdir(logs)), f"Missing results for {label}: {logs}"
        if policy in ("token", "optimal"):
            label = bucket_label(label, run)
        srcs.append(logs)
        labels.append(label)

    fig_dir = os.path.join(RESULTS, core_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_tail = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_tail.py")
    variants = [(NORMALIZED[family], f"{name}_{workload}_tail.png")]
    if family in MS_PLOTS:
        variants.append(([], f"{name}_{workload}_tail_ms.png"))

    for options, out in variants:
        subprocess.run([sys.executable, plot_tail, "--pdf"] + options
                       + ["--srcs"] + srcs + ["--labels"] + labels
                       + ["--out", os.path.join(fig_dir, out)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="+", choices=[f for _, f, _ in CORE_RUNS],
                        default=[f for _, f, _ in CORE_RUNS],
                        help="Families to plot (default: all)")
    args = parser.parse_args()

    for core_dir, family, name in CORE_RUNS:
        if family in args.families:
            for workload in ("const", "burst"):
                plot(core_dir, family, name, workload)
