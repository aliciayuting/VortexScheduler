"""Plots response time over the course of each run for the policies the RR-drop
tail CDFs compare (plot_rrdrop_tails.py), one panel per policy.

Each panel puts every completed job at its arrival time and response time, and
marks the arrival time of every job the policy dropped or rejected with a tick
along the panel's top edge, so it shows when in the run a policy sheds work and
what the jobs it keeps wait. The panels share both axes: the y axis is log
scaled, since a run without shedding waits orders of magnitude longer than one
with it. A dashed line marks the SLO, and bursty runs shade the spikes their
workload config schedules.

Writes, into [src]/figures/:
    <name>_<workload>_response_over_time.png

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_rrdrop_over_time.py \\
        --src experiments/results/rrdrop_min_unserved --families wf6
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_rrdrop_tails as prt
from plot_queue_lengths import SPIKE_SHADE, spike_windows
from plot_tail import SLO_STYLE
from run_rrdrop_sweep import CORE_DIRS, NAMES

DROP_TICK = "0.55"
# text sizes (pt), larger than the tail CDFs' since the figure is tall and is
# scaled down further to fit a column
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
TITLE_FONTSIZE = 16
LEGEND_FONTSIZE = 16
PANEL_HEIGHT = 2.0


def _jobs(logs: str) -> pd.DataFrame:
    jobs = pd.read_csv(os.path.join(logs, "job_log.csv"))
    jobs["completed"] = jobs["was_completed"].astype(bool)
    jobs["slo"] = jobs["deadline"] - jobs["create_time"]
    return jobs


def plot(src: str, family: str, workload: str):
    logs, labels = prt.policy_runs(src, family, workload)
    colors = list(sns.color_palette("tab10", len(logs)))
    runs = [_jobs(d) for d in logs]
    spikes = spike_windows(os.path.dirname(logs[0]))

    slos = np.unique(np.concatenate([j["slo"].round(6).unique() for j in runs]))
    done = np.concatenate([j.loc[j["completed"], "response_time"].values for j in runs])
    t_end = max(j["create_time"].max() for j in runs) / 1000

    fig, axes = plt.subplots(len(runs), 1, sharex=True, sharey=True,
                             figsize=(12, PANEL_HEIGHT * len(runs) + 1.2))
    for ax, jobs, label, color in zip(axes, runs, labels, colors):
        for start, end in spikes:
            ax.axvspan(start, end, color=SPIKE_SHADE, linewidth=0, zorder=0)

        kept = jobs[jobs["completed"]]
        ax.scatter(kept["create_time"] / 1000, kept["response_time"], s=2,
                   color=color, alpha=0.35, linewidths=0, rasterized=True, zorder=2)

        shed = jobs.loc[~jobs["completed"], "create_time"] / 1000
        # a tick per shed job along the top edge, which reads as a band where
        # shedding is dense
        ax.vlines(shed, 0.9, 1.0, transform=ax.get_xaxis_transform(), color=DROP_TICK,
                  linewidth=0.3, rasterized=True, zorder=1)

        for slo in slos:
            ax.axhline(slo, **SLO_STYLE)

        met = (jobs["completed"] & (jobs["response_time"] <= jobs["slo"])).mean()
        ax.set_title(label, loc="left", fontsize=TITLE_FONTSIZE)
        ax.set_title(f"SLO met {100 * met:.1f}%, {100 * (1 - jobs['completed'].mean()):.1f}% dropped",
                     loc="right", fontsize=TITLE_FONTSIZE, color="0.3")
        ax.set_yscale("log")
        ax.grid(True, which="major", color="0.88", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(which="both", labelsize=TICK_FONTSIZE)

    # headroom above the slowest job for the shed-job ticks
    axes[0].set_ylim(done.min() * 0.8, done.max() * 2.5)
    axes[0].set_xlim(0, t_end)
    axes[-1].set_xlabel("Arrival time (s)", fontsize=LABEL_FONTSIZE)
    fig.supylabel("Response time (ms)", fontsize=LABEL_FONTSIZE)

    handles = [Line2D([], [], marker="o", linestyle="", color="0.35", markersize=5,
                      label="completed job"),
               Line2D([], [], marker="|", linestyle="", color=DROP_TICK, markersize=10,
                      markeredgewidth=1.5, label="dropped or rejected job"),
               Line2D([], [], label="SLO", **SLO_STYLE)]
    if spikes:
        handles.append(Patch(color=SPIKE_SHADE, label="spike in send rate"))

    fig.tight_layout()
    fig.legend(handles=handles, loc="upper right", ncol=len(handles),
               fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(axes[-1].get_position().x1, 0),
               frameon=False)

    out = os.path.join(src, "figures", f"{NAMES[family]}_{workload}_response_over_time.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=str, required=True,
                        help="run_rrdrop_sweep.py results directory")
    parser.add_argument("--families", nargs="+", default=sorted({f for f, _ in CORE_DIRS}),
                        help="Families to plot (default: all)")
    args = parser.parse_args()

    for family, workload in CORE_DIRS:
        if family in args.families:
            plot(args.src, family, workload)
