"""Plots how full the token bucket ran over each refill rate sweep run, and
summarizes its level.

The run's token_bucket_log.csv records the bucket's level at each admission
decision. Between decisions the bucket refills linearly at its rate, capped at its
depth, so the level is reconstructed exactly on a 1 ms grid from the level each
decision left it at.

Writes, into [src]/figures/token_levels/:
    <family>_<workload>.png  tokens in the bucket over time, one panel per refill rate
and into [src]/figures/:
    token_stats.png          time-averaged fill and share of time empty against
                             refill rate, one column per workflow
and into [src]/figures/token_level_hist/:
    <family>.png             histogram of tokens in the bucket over time, a row per
                             refill rate and a column per workload
and [src]/token_stats.csv with one row per run.

Reads the results of experiments/run_refill_rate_sweep.py.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_token_levels.py \
        --src experiments/results/refill_rate_sweep
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_bucket_size_sweep import (FAMILIES, SERIES, INK, INK_SECONDARY, SURFACE,
                                    _style)
from plot_refill_rate_sweep import RATES, _run_dir

# resolution the bucket level is reconstructed at (ms)
GRID_MS = 1.0
# the level plots draw the range the level spans in each bin of this width (ms),
# so short dips to empty stay visible
PLOT_BIN_MS = 20
# bins the level's histogram splits the bucket's depth into
HIST_BINS = 40


def bucket_level(run_dir: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    """Returns the bucket's level sampled every GRID_MS from time 0 to the last
    decision, the run's decisions, and the bucket's refill rate (tokens/ms) and depth.
    """
    log = pd.read_csv(os.path.join(run_dir, "sim_logs", "token_bucket_log.csv"))
    adm = pd.read_csv(os.path.join(run_dir, "sim_logs", "admission_control.csv"))
    assert(log["bucket"].nunique() == 1 and len(adm) == 1), \
        f"{run_dir}: expected a single bucket"

    rate = adm["admit_rate_qps"].iloc[0] / 1000
    depth = log["depth"].iloc[0]
    times = log["time"].to_numpy()
    left = log["tokens_left"].to_numpy()

    grid = np.arange(0, times[-1] + GRID_MS, GRID_MS)
    # the last decision at or before each grid point; the bucket starts full
    last = np.searchsorted(times, grid, side="right") - 1
    level = np.where(last < 0, depth,
                     np.minimum(depth, left[np.maximum(last, 0)]
                                + rate * (grid - times[np.maximum(last, 0)])))

    return grid, level, log, {"rate": rate, "depth": depth}


def token_stats(src: str) -> tuple[pd.DataFrame, dict, dict]:
    """Returns one row of bucket level statistics per run, and for plotting each
    run's binned level over time and the share of time it spent in each of
    HIST_BINS equal bins of its depth, both keyed by (family, workload, rate).
    """
    rows, traces, hists = [], {}, {}
    for family, _ in FAMILIES:
        for workload, _, _ in SERIES:
            for rate in RATES:
                run = _run_dir(src, family, workload, rate)
                if not os.path.isfile(os.path.join(run, "sim_logs", "token_bucket_log.csv")):
                    continue

                grid, level, log, bucket = bucket_level(run)
                depth = bucket["depth"]
                n = len(level) // int(PLOT_BIN_MS / GRID_MS) * int(PLOT_BIN_MS / GRID_MS)
                bins = level[:n].reshape(-1, int(PLOT_BIN_MS / GRID_MS))
                traces[(family, workload, rate)] = (
                    grid[:n:int(PLOT_BIN_MS / GRID_MS)], bins.min(axis=1), bins.max(axis=1),
                    depth)
                # the last bin includes depth itself, so time spent full lands there
                counts, edges = np.histogram(level, bins=np.linspace(0, depth, HIST_BINS + 1))
                hists[(family, workload, rate)] = (counts / len(level), edges)
                rows.append({
                    "family": family, "workload": workload, "rate": rate,
                    "refill_qps": 1000 * bucket["rate"], "depth": depth,
                    "duration_s": grid[-1] / 1000,
                    # time-weighted over the run
                    "mean_tokens": level.mean(),
                    "mean_fill": level.mean() / depth,
                    "p5_tokens": np.percentile(level, 5),
                    "p50_tokens": np.percentile(level, 50),
                    "p95_tokens": np.percentile(level, 95),
                    "min_tokens": log["tokens_left"].min(),
                    # under one token, an arriving job is rejected
                    "time_empty": (level < 1).mean(),
                    # at depth, refill is discarded
                    "time_full": (level >= depth - 1e-9).mean(),
                    # as seen by arriving jobs
                    "mean_tokens_at_arrival": log["tokens_available"].mean(),
                    "rejected": 1 - log["admitted"].mean(),
                })

    return pd.DataFrame(rows), traces, hists


def plot_levels(stats: pd.DataFrame, traces: dict, fig_dir: str, default_rate: float):
    """Tokens in the bucket over time for one workflow and workload, a panel per
    refill rate sharing both axes."""
    for family, name in FAMILIES:
        for workload, wl_label, color in SERIES:
            rates = [r for r in RATES if (family, workload, r) in traces]
            if not rates:
                continue

            fig, axes = plt.subplots(len(rates), 1, figsize=(10, 1.15 * len(rates) + 0.9),
                                     sharex=True, sharey=True, squeeze=False)
            fig.patch.set_facecolor(SURFACE)
            for ax, rate in zip(axes[:, 0], rates):
                start, low, high, depth = traces[(family, workload, rate)]
                row = stats[(stats["family"] == family) & (stats["workload"] == workload)
                            & (stats["rate"] == rate)].iloc[0]
                _style(ax, "")
                ax.axhline(depth, color=INK_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
                ax.fill_between(start / 1000, low, high, color=color, linewidth=0.4,
                                edgecolor=color, step="post")
                ax.set_ylim(0, depth * 1.08)
                ax.set_ylabel(f"refill {rate:g}" + ("\n(default)" if np.isclose(rate, default_rate) else ""),
                              color=INK, fontsize=9, rotation=0, ha="right", va="center")
                ax.text(1.0, 1.0, f"mean {100 * row['mean_fill']:.0f}% full · "
                                  f"empty {100 * row['time_empty']:.1f}% of time · "
                                  f"rejected {100 * row['rejected']:.1f}%",
                        transform=ax.transAxes, ha="right", va="bottom",
                        color=INK_SECONDARY, fontsize=8)

            axes[0, 0].set_title(f"{name}, {wl_label}: tokens in the bucket "
                                 f"(depth {depth:.1f} jobs, dashed)",
                                 color=INK, fontsize=11, loc="left", pad=16)
            axes[-1, 0].set_xlabel("Time (s)", color=INK_SECONDARY, fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"{family}_{workload}.png"), dpi=200,
                        facecolor=SURFACE)
            plt.close(fig)


def plot_stats(stats: pd.DataFrame, out_path: str, default_rate: float):
    """Time-averaged fill (top) and share of time the bucket is empty (bottom)
    against refill rate, one column per workflow."""
    families = [(f, n) for f, n in FAMILIES if (stats["family"] == f).any()]
    rates = [r for r in RATES if (stats["rate"] == r).any()]

    fig, axes = plt.subplots(2, len(families), figsize=(4 * len(families), 7.5),
                             sharex=True, sharey="row", squeeze=False)
    for col, (family, name) in enumerate(families):
        rows = stats[stats["family"] == family]
        _style(axes[0][col], f"{name}\nbucket: {rows['depth'].iloc[0]:.1f} jobs")
        _style(axes[1][col], "")

        for row, (metric, ylabel) in enumerate(
                [("mean_fill", "Mean bucket level (% of depth)"),
                 ("time_empty", "Time the bucket is empty (%)")]):
            ax = axes[row][col]
            ax.axvline(default_rate, color=INK_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
            ax.axvline(1.0, color=INK_SECONDARY, linewidth=1, linestyle=(0, (1, 2)))
            for workload, label, color in SERIES:
                pts = rows[rows["workload"] == workload].sort_values("rate")
                ax.plot(pts["rate"], 100 * pts[metric], label=label, color=color,
                        linewidth=2, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=2)

            ax.set_xticks(rates)
            ax.set_xticklabels([f"{r:g}" if i % 2 == 0 else "" for i, r in enumerate(rates)])
            if col == 0:
                ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
            if row == 1:
                ax.set_xlabel("Refill rate (fraction of estimated capacity)",
                              color=INK_SECONDARY, fontsize=10)

    ax = axes[0][0]
    ax.annotate("default", (default_rate, 1), xycoords=("data", "axes fraction"),
                xytext=(-3, -3), textcoords="offset points", ha="right", va="top",
                color=INK_SECONDARY, fontsize=8)
    ax.annotate("capacity\nestimate", (1.0, 1), xycoords=("data", "axes fraction"),
                xytext=(3, -3), textcoords="offset points", ha="left", va="top",
                color=INK_SECONDARY, fontsize=8)
    axes[0][0].set_ylim(0, 100)
    axes[1][0].set_ylim(0, None)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_hists(stats: pd.DataFrame, hists: dict, fig_dir: str, default_rate: float):
    """Histogram of tokens in the bucket over time for one workflow, a row per
    refill rate and a column per workload, sharing both axes."""
    for family, name in FAMILIES:
        rates = [r for r in RATES if any((family, w, r) in hists for w, _, _ in SERIES)]
        if not rates:
            continue

        fig, axes = plt.subplots(len(rates), len(SERIES),
                                 figsize=(5.2 * len(SERIES) + 1.2, 1.15 * len(rates) + 1.2),
                                 sharex=True, sharey=True, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for col, (workload, wl_label, color) in enumerate(SERIES):
            for row, rate in enumerate(rates):
                ax = axes[row][col]
                _style(ax, wl_label if row == 0 else "")
                if (family, workload, rate) not in hists:
                    continue
                shares, edges = hists[(family, workload, rate)]
                run = stats[(stats["family"] == family) & (stats["workload"] == workload)
                            & (stats["rate"] == rate)].iloc[0]

                # below one token an arriving job is rejected
                ax.axvline(1, color=INK_SECONDARY, linewidth=1, linestyle=(0, (1, 2)))
                ax.stairs(100 * shares, edges, fill=True, color=color, linewidth=0)
                ax.text(1.0, 1.0, f"mean {run['mean_tokens']:.1f} · "
                                    f"empty {100 * run['time_empty']:.1f}% · "
                                    f"full {100 * run['time_full']:.1f}%",
                        transform=ax.transAxes, ha="right", va="bottom",
                        color=INK_SECONDARY, fontsize=8)
                if col == 0:
                    ax.set_ylabel(f"refill {rate:g}"
                                  + ("\n(default)" if np.isclose(rate, default_rate) else ""),
                                  color=INK, fontsize=9, rotation=0, ha="right", va="center")

            depth = stats.loc[stats["family"] == family, "depth"].iloc[0]
            axes[-1][col].set_xlim(0, depth)
            axes[-1][col].set_xlabel("Tokens in the bucket", color=INK_SECONDARY, fontsize=10)
        axes[0][0].annotate("1 token", (1, 1), xycoords=("data", "axes fraction"),
                            xytext=(3, -3), textcoords="offset points", ha="left", va="top",
                            color=INK_SECONDARY, fontsize=8)

        fig.suptitle(f"{name}: share of time (%) at each bucket level "
                     f"(depth {depth:.1f} jobs, {HIST_BINS} bins; time full lands in the last)",
                     color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{family}.png"), dpi=200, facecolor=SURFACE)
        plt.close(fig)


if __name__ == "__main__":
    import core.configs.gen_config as gcfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=str, required=True, help="Sweep results directory")
    args = parser.parse_args()

    level_dir = os.path.join(args.src, "figures", "token_levels")
    os.makedirs(level_dir, exist_ok=True)

    stats, traces, hists = token_stats(args.src)
    stats.to_csv(os.path.join(args.src, "token_stats.csv"), index=False)
    plot_stats(stats, os.path.join(args.src, "figures", "token_stats.png"),
               gcfg.ADMISSION_TARGET_UTILIZATION)
    plot_levels(stats, traces, level_dir, gcfg.ADMISSION_TARGET_UTILIZATION)
    hist_dir = os.path.join(args.src, "figures", "token_level_hist")
    os.makedirs(hist_dir, exist_ok=True)
    plot_hists(stats, hists, hist_dir, gcfg.ADMISSION_TARGET_UTILIZATION)
    print(f"wrote {args.src}/token_stats.csv and {args.src}/figures/")
