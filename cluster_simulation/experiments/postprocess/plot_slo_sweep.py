"""Plots a drop policy comparison across a sweep of job SLOs.

A tail CDF answers "how slow were the jobs of one run"; this answers "how does the
policy hold up as the job SLO grows", which a CDF cannot show on one axis once the
runs being compared have different SLOs. The tail statistic plotted is the share
of jobs that miss the job SLO, i.e. the tail measured at the deadline itself,
counting jobs that were dropped or never finished. A response time percentile
would not do: once more than 1% of jobs are dropped, p99 is infinite.

Reads the results of experiments/run_lookahead_sweep.py.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_slo_sweep.py \
        --src experiments/results/lookahead_queue_delay
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# job size (ms) per workflow: its critical path at batch size 1, which the job SLO
# is a multiple of
JOB_SIZES = {1: 186.3, 6: 62.48}

# categorical slots 1 and 2, carrying the comparison itself
SERIES = [("ahead", "LOOKAHEAD (ahead only)", "#2a78d6"),
          ("whole", "LOOKAHEAD (whole queue)", "#eb6834")]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"


def _miss_rate(run_dir: str, slo: float) -> float:
    """Returns the percentage of jobs that did not finish within [slo], counting
    the jobs that never finished at all.
    """
    jobs = pd.read_csv(os.path.join(run_dir, "sim_logs", "job_log.csv"))

    return 100 * (1 - (jobs["was_completed"] & (jobs["response_time"] <= slo)).mean())


def plot_slo_sweep(src: str, multiples: list[int], out_path: str):
    """Plots missed SLO share against job SLO, one panel per workflow and workload.

    Args:
        src: Sweep results directory, holding one directory per configuration
        multiples: Job SLOs to plot, as multiples of job size
        out_path: Where to write the figure
    """
    panels = [(wid, workload) for wid in sorted(JOB_SIZES)
              for workload in ("const", "burst")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

    for ax, (wid, workload) in zip(axes.flat, panels):
        for key, label, color in SERIES:
            xs, ys = [], []
            for mult in multiples:
                run_dir = os.path.join(src, f"wf{wid}_{workload}_{mult}x_{key}")
                if not os.path.isdir(run_dir):
                    continue

                xs.append(mult)
                ys.append(_miss_rate(run_dir, int(JOB_SIZES[wid] * mult)))

            ax.plot(xs, ys, label=label, color=color, linewidth=2,
                    marker="o", markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=2)

        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        ax.set_title(f"WF{wid}, {workload}", color=INK, fontsize=11, loc="left")

    axes[0][0].set_ylim(0, None)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    for ax in axes[1]:
        ax.set_xlabel("Job SLO (multiples of job size)", color=INK_SECONDARY,
                      fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Jobs missing the SLO (%)", color=INK_SECONDARY, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, required=True,
                        help="Sweep results directory")
    parser.add_argument("--multiples", type=int, nargs="+",
                        default=[3, 4, 5, 6, 7, 8, 9, 10],
                        help="Job SLOs to plot, as multiples of job size")
    parser.add_argument("--out", type=str,
                        help="Output path (default: [src]/figures/slo_sweep.png)")

    args = parser.parse_args()

    out_path = args.out or os.path.join(args.src, "figures", "slo_sweep.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plot_slo_sweep(args.src, args.multiples, out_path)
    print(f"wrote {out_path}")
