"""Plots how many tasks were queued for each model over the course of a run.

Reads each run's queue_length_log.csv, which records a worker queue's length every
time it changes, and sums the queues of every worker hosting a model into that
model's backlog. The backlog moves on every arrival and every batch, so it is
plotted as its time-weighted mean over fixed bins rather than point by point.
Bursty runs shade the spikes their workload config schedules.

Writes a grid with a row per model and a column per run, so each panel holds a
single backlog. Each panel keeps its own y axis, since backlogs differ by orders
of magnitude between models and runs. With --per-instance, each panel instead
draws every instance's own queue (thin) and their mean across instances (bold),
which shows how evenly dispatch spreads the model's work.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_queue_lengths.py \\
        --srcs experiments/results/wf14_core/wf14_const_nodrop \\
               experiments/results/wf14_core/wf14_burst_nodrop \\
        --labels "WF14, constant" "WF14, bursty" --out queue_lengths.png
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_bucket_size_sweep import INK_SECONDARY, SURFACE, _style

# categorical slots in fixed order, assigned to models by ascending model ID
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]
SPIKE_SHADE = "#dedcd6"


def _model_names(run_dir: str) -> dict[int, str]:
    """Returns model ID -> name from the run's own config snapshot."""
    spec = importlib.util.spec_from_file_location(
        "snapshot_models", os.path.join(run_dir, "configs", "model_config.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {i: m.get("MODEL_NAME", f"model {i}") for i, m in enumerate(module.MODELS)}


def _read_log(run_dir: str, bin_ms: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns the run's queue length log in time order, and the edges (ms) of
    [bin_ms] wide bins spanning it."""
    log = pd.read_csv(os.path.join(run_dir, "sim_logs", "queue_length_log.csv"))
    log = log.sort_values("time", kind="stable")
    return log, np.arange(0, log["time"].max() + bin_ms, bin_ms)


def _binned_mean(times: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Returns the time-weighted mean, over each bin between [edges], of a step
    function that takes [values][k] from [times][k] until the next change (and 0
    before the first)."""
    # area under the step function up to each change, then up to each bin edge
    area = np.concatenate([[0.0], np.cumsum(values[:-1] * np.diff(times))])
    i = np.searchsorted(times, edges, side="right") - 1
    held = np.where(i >= 0, values[np.clip(i, 0, None)], 0.0)
    area_at = np.where(i >= 0, area[np.clip(i, 0, None)]
                       + held * (edges - times[np.clip(i, 0, None)]), 0.0)
    return np.diff(area_at) / np.diff(edges)


def model_backlogs(run_dir: str, bin_ms: float) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Returns the bin start times (s) and, per model, the time-weighted mean
    number of tasks queued for it in each bin, summed over workers.

    Args:
        run_dir: One run's results directory
        bin_ms: Width of the averaging bins (ms)
    """
    log, edges = _read_log(run_dir, bin_ms)

    backlogs = {}
    for model_id, rows in log.groupby("model_id"):
        # each row sets one worker's queue; the model's backlog moves by the change
        lengths = rows["queue_length"].to_numpy()
        previous = rows.groupby("worker_id")["queue_length"].shift(fill_value=0).to_numpy()
        backlogs[model_id] = _binned_mean(rows["time"].to_numpy(),
                                          np.cumsum(lengths - previous), edges)

    return edges[:-1] / 1000, backlogs


def instance_backlogs(run_dir: str, bin_ms: float) -> tuple[np.ndarray,
                                                            dict[int, list[np.ndarray]]]:
    """Returns the bin start times (s) and, per model, the time-weighted mean
    length of each instance's own queue in each bin. A worker holds one queue per
    model it hosts, so each worker hosting a model is one of its instances.
    """
    log, edges = _read_log(run_dir, bin_ms)

    backlogs: dict[int, list[np.ndarray]] = {}
    for (model_id, _), rows in log.groupby(["model_id", "worker_id"]):
        backlogs.setdefault(model_id, []).append(
            _binned_mean(rows["time"].to_numpy(), rows["queue_length"].to_numpy(), edges))

    return edges[:-1] / 1000, backlogs


def spike_windows(run_dir: str) -> list[tuple[float, float]]:
    """Returns the (start, end) times (s) of every spike the run's workload
    schedules. BURST phases start quiet; a stochastic one has no fixed schedule
    and is left unshaded.
    """
    with open(os.path.join(run_dir, "overrides.json")) as f:
        clients = json.load(f)["overrides"].get("CLIENT_CONFIGS", [])

    windows = []
    for client in clients:
        for cfg in client.values():
            start = 0.0
            for phase in cfg["WORKLOAD"]["PHASES"]:
                if phase.get("KIND") == "BURST" and not phase.get("STOCHASTIC", True):
                    quiet, burst = phase["QUIET_DURATION"], phase["BURST_DURATION"]
                    t = start + quiet
                    while t < start + phase["DURATION"]:
                        windows.append((t / 1000, min(t + burst, start + phase["DURATION"]) / 1000))
                        t += quiet + burst
                start += phase.get("DURATION", 0)

    return windows


def plot_queue_lengths(srcs: list[str], labels: list[str], bin_ms: float,
                       out_path: str, per_instance: bool = False):
    runs = [(instance_backlogs if per_instance else model_backlogs)(src, bin_ms)
            for src in srcs]
    names = {}
    for src in srcs:
        names.update(_model_names(src))

    # a row per model, in ascending model ID order, each in its own color
    ordered = sorted(set().union(*(backlogs.keys() for _, backlogs in runs)))
    colors = {m: SERIES_COLORS[k % len(SERIES_COLORS)] for k, m in enumerate(ordered)}

    fig, axes = plt.subplots(len(ordered), len(srcs),
                             figsize=(6.5 * len(srcs), 2.8 * len(ordered)),
                             sharex="col", squeeze=False)

    for col, (src, label, (times, backlogs)) in enumerate(zip(srcs, labels, runs)):
        spikes = spike_windows(src)
        for row, model_id in enumerate(ordered):
            ax = axes[row][col]
            _style(ax, f"{label}: {names.get(model_id, model_id)} (model {model_id})")

            for start, end in spikes:
                ax.axvspan(start, end, color=SPIKE_SHADE, linewidth=0, zorder=0)

            if model_id in backlogs and per_instance:
                for queue in backlogs[model_id]:
                    ax.plot(times, queue, linewidth=0.8, alpha=0.35, color=colors[model_id])
                ax.plot(times, np.mean(backlogs[model_id], axis=0), linewidth=1.8,
                        color=colors[model_id])
            elif model_id in backlogs:
                ax.plot(times, backlogs[model_id], linewidth=1.5, color=colors[model_id])
            ax.set_ylim(0, None)
            ax.set_xlim(0, times[-1])

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)", color=INK_SECONDARY, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(("Tasks queued per instance" if per_instance else "Tasks queued")
                      + f"\n(mean per {bin_ms / 1000:g} s)",
                      color=INK_SECONDARY, fontsize=10)

    # each panel names its own model, so only the line weights and shading need a key
    handles = []
    if per_instance:
        handles += [plt.Line2D([], [], color=INK_SECONDARY, linewidth=0.8, alpha=0.5,
                               label="one instance's queue"),
                    plt.Line2D([], [], color=INK_SECONDARY, linewidth=1.8,
                               label="mean over the model's instances")]
    if any(spike_windows(src) for src in srcs):
        handles.append(Patch(color=SPIKE_SHADE, label="spike in send rate"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   fontsize=9, labelcolor=INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--srcs", nargs="+", required=True,
                        help="Run directories, each holding sim_logs/queue_length_log.csv")
    parser.add_argument("--labels", nargs="+", required=True, help="Panel title per run")
    parser.add_argument("--bin-ms", type=float, default=1000,
                        help="Width of the averaging bins (ms, default 1000)")
    parser.add_argument("--per-instance", action="store_true",
                        help="Plot each instance's queue and their mean, rather than "
                             "each model's total backlog")
    parser.add_argument("--out", type=str, required=True, help="Output image path")
    args = parser.parse_args()

    assert(len(args.srcs) == len(args.labels)), "one label per source"
    plot_queue_lengths(args.srcs, args.labels, args.bin_ms, args.out, args.per_instance)
    print(f"wrote {args.out}")
