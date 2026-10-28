"""Plots a drop policy comparison across a sweep of job SLOs.

A tail CDF answers "how slow were the jobs of one run"; this answers "how does the
policy hold up as the job SLO grows", which a CDF cannot show on one axis once the
runs being compared have different SLOs. The tail statistic plotted is the share
of jobs that miss the job SLO, i.e. the tail measured at the deadline itself,
counting jobs that were dropped or never finished. A response time percentile
would not do: once more than 1% of jobs are dropped, p99 is infinite.

Reads the results of experiments/run_lookahead_sweep.py, one queue ordering at a
time. Each run holds every tenant of its tenancy, so there is one row per tenant.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_slo_sweep.py \
        --src experiments/results/lookahead_multitenant --ordering fcfs
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# tenancies run_lookahead_sweep.py names its configurations after, in plot order
TENANCIES = ["ppl2", "textvision", "textvisionswapped",
             "wf1", "wf6", "wf12", "wf13", "wf14"]

# single-tenant tenancies whose SLO multiples are of another workflow's job size,
# as run_core_sweep.SLO_FROM sets them
SLO_FROM_LABELS = {"wf14": "WF12's deadlines"}

# categorical slots 1 and 2, carrying the comparison itself
SERIES = [("ahead", "LOOKAHEAD (ahead only)", "#2a78d6"),
          ("whole", "LOOKAHEAD (whole queue)", "#eb6834")]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"


def _run_dir(src: str, tenancy: str, workload: str, mult: int, ordering: str,
             accounting: str) -> str:
    return os.path.join(src, f"{tenancy}_{workload}_{mult}x_{ordering}_{accounting}")


def _read_jobs(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "sim_logs", "job_log.csv"))


def _miss_rate(jobs: pd.DataFrame, wid: int) -> float:
    """Returns the percentage of [wid]'s jobs that did not finish within the job
    SLO the run gave them, counting the jobs that never finished at all.

    The SLO is read back from each job's own deadline rather than recomputed from
    a job size, so it is always the one the run was actually configured with.
    """
    jobs = jobs[jobs["workflow_id"] == wid]
    met = jobs["was_completed"].astype(bool) \
        & (jobs["response_time"] <= jobs["deadline"] - jobs["create_time"])

    return 100 * (1 - met.mean())


def plot_slo_sweep(src: str, multiples: list[int], ordering: str, out_path: str):
    """Plots missed SLO share against job SLO, one panel per tenant and workload.

    Args:
        src: Sweep results directory, holding one directory per configuration
        multiples: Job SLOs to plot, as multiples of job size
        ordering: Queue ordering to plot, as named in the configurations
        out_path: Where to write the figure
    """
    run_dirs = {(tenancy, workload, mult, key): _run_dir(src, tenancy, workload, mult,
                                                          ordering, key)
                for tenancy in TENANCIES for workload in ("const", "burst")
                for mult in multiples for key, _, _ in SERIES}
    run_dirs = {k: d for k, d in run_dirs.items() if os.path.isdir(d)}
    assert(run_dirs), f"No {ordering} runs of run_lookahead_sweep.py found in {src}"

    jobs = {k: _read_jobs(d) for k, d in run_dirs.items()}

    # one row per tenant of every tenancy the sweep actually ran
    rows = sorted({(tenancy, wid) for (tenancy, *_), df in jobs.items()
                   for wid in df["workflow_id"].unique()},
                  key=lambda r: (TENANCIES.index(r[0]), r[1]))
    panels = [(tenancy, wid, workload) for tenancy, wid in rows
              for workload in ("const", "burst")]

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 4 * len(rows)),
                             sharex=True, sharey=True, squeeze=False)

    for ax, (tenancy, wid, workload) in zip(axes.flat, panels):
        for key, label, color in SERIES:
            xs, ys = [], []
            for mult in multiples:
                if (tenancy, workload, mult, key) not in jobs:
                    continue

                xs.append(mult)
                ys.append(_miss_rate(jobs[(tenancy, workload, mult, key)], wid))

            # a series the sweep did not run is left out of the legend too
            if not xs:
                continue

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
        # a single-tenant tenancy is named after its workflow, so only say which
        # tenancy a panel is from when it is shared, or held to another's SLO
        note = SLO_FROM_LABELS.get(tenancy) or (tenancy if tenancy != f"wf{wid}" else None)
        ax.set_title(f"WF{wid}" + (f" ({note})" if note else "") + f", {workload}",
                     color=INK, fontsize=11, loc="left")

    axes[0][0].set_ylim(0, None)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    for ax in axes[-1]:
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
    parser.add_argument("--ordering", type=str, default="fcfs",
                        choices=["fcfs", "edf", "ap"],
                        help="Queue ordering to plot (default: fcfs)")
    parser.add_argument("--out", type=str,
                        help="Output path "
                             "(default: [src]/figures/slo_sweep_[ordering].png)")

    args = parser.parse_args()

    out_path = args.out or os.path.join(args.src, "figures",
                                        f"slo_sweep_{args.ordering}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plot_slo_sweep(args.src, args.multiples, args.ordering, out_path)
    print(f"wrote {out_path}")
