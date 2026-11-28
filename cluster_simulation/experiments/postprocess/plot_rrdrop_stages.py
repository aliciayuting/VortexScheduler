"""Plots, for the policies the RR-drop tail CDFs compare (plot_rrdrop_tails.py),
how large the batches each model executed were and at which stage jobs were shed.

Batch sizes come from each run's worker_batch_log.csv. A model's histogram gives,
for each batch size, the share of the tasks it executed that ran in a batch of
that size, i.e. the share of jobs whose task at that stage did, counting a job
once per stage when several of its stages run the same model. Models that never
batch (a maximum batch size of 1) are left out.

Drop stages come from the job log, for the policies that drop jobs in flight
(admission control rejects jobs before they reach any stage, and is left out). A
dropped job is attributed to the model whose queue it was dropped from, which the job log
records as drop_model_id; stages that run the same model are one stage. Runs
from before that column existed are read from [stage_src] instead, a rerun of the
drop policies with it, after checking that the rerun completed and dropped
exactly the same jobs as the run the tail CDFs draw.

Writes, into [src]/figures/:
    <name>_<workload>_batch_sizes.png
    <name>_<workload>_drop_stages.png

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/postprocess/plot_rrdrop_stages.py \\
        --src experiments/results/rrdrop_min_unserved --families wf13 \\
        --stage-src experiments/results/wf13_dropstage
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_rrdrop_tails as prt
from run_rrdrop_sweep import CORE_DIRS, NAMES

LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
TITLE_FONTSIZE = 17
# the drop stage figure is a single short row, so it can carry larger text
DROP_LABEL_FONTSIZE = 20
DROP_TICK_FONTSIZE = 16
DROP_TITLE_FONTSIZE = 17
DROP_BAR_FONTSIZE = 14

# short names for the models in the figures
MODEL_NAMES = {20: "cow ID OCR", 21: "teat segmentation",
               22: "teat shape", 23: "teat skin condition",
               24: "aggregation"}


def _short(label: str) -> str:
    """The policy name of a legend label, without its parameters."""
    return label.split(" (")[0]


def _jobs(logs: str) -> pd.DataFrame:
    jobs = pd.read_csv(os.path.join(logs, "job_log.csv"))
    jobs["completed"] = jobs["was_completed"].astype(bool)
    return jobs.sort_values("job_id").reset_index(drop=True)


def _drop_policy(run: str) -> str:
    """The DROP_POLICY [run] ran with: the last assignment in its config copy,
    which run scripts append their overrides to."""
    with open(os.path.join(run, "configs", "gen_config.py")) as f:
        values = [line.split("=", 1)[1].strip().strip("\"'") for line in f
                  if line.startswith("DROP_POLICY =")]
    assert(values), f"no DROP_POLICY in {run}'s config"
    return values[-1]


def _stage_logs(logs: str, stage_src: str | None) -> str:
    """The sim_logs directory to read [logs]' drop stages from: [logs] itself if
    its job log has drop_model_id or it never dropped from a queue, otherwise the
    rerun of the same configuration in [stage_src], once checked to reproduce it.
    """
    jobs = _jobs(logs)
    if ("drop_model_id" in jobs or jobs["completed"].all()
            or _drop_policy(os.path.dirname(logs)) == "NONE"):
        return logs

    run = os.path.basename(os.path.dirname(logs))
    rerun = os.path.join(stage_src or "", run, "sim_logs")
    assert(stage_src and os.path.isdir(rerun)), \
        f"{run} predates drop_model_id and has no rerun in {stage_src}"
    staged = _jobs(rerun)
    assert(staged["job_id"].equals(jobs["job_id"])
           and staged["completed"].equals(jobs["completed"])
           and np.allclose(staged["response_time"].fillna(-1), jobs["response_time"].fillna(-1))), \
        f"the rerun of {run} in {stage_src} did not reproduce it"
    return rerun


def _stage_jobs(logs: str) -> pd.DataFrame:
    """The job log of [logs] (from [_stage_logs]) with drop_model_id. A run that
    never dropped from a queue has none to log: every job it shed was rejected
    at admission."""
    jobs = _jobs(logs)
    if "drop_model_id" not in jobs:
        jobs["drop_model_id"] = np.nan
    return jobs


def _style(ax, ticksize: float = TICK_FONTSIZE):
    ax.grid(True, axis="y", color="0.88", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=ticksize)


def plot_batch_sizes(logs: list[str], labels: list[str], colors: list, out: str):
    batches = [pd.read_csv(os.path.join(d, "worker_batch_log.csv")) for d in logs]
    largest = pd.concat(batches).groupby("model_id")["batch_size"].max()
    models = [m for m in sorted(largest.index) if largest[m] > 1]

    fig, axes = plt.subplots(len(logs), len(models), sharey=True, squeeze=False,
                             figsize=(3.6 * len(models) + 1.5, 1.9 * len(logs) + 1.2))
    for row, (batch, label, color) in enumerate(zip(batches, labels, colors)):
        for col, model in enumerate(models):
            ax = axes[row][col]
            sizes = batch.loc[batch["model_id"] == model, "batch_size"]
            # a batch of size b executes b tasks
            tasks = sizes.value_counts().reindex(range(1, largest[model] + 1), fill_value=0) \
                * np.arange(1, largest[model] + 1)
            ax.bar(tasks.index, tasks / tasks.sum(), color=color, width=0.8)
            ax.set_xticks(range(1, largest[model] + 1))
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
            _style(ax)
            if row == 0:
                ax.set_title(MODEL_NAMES.get(model, f"model {model}"), fontsize=TITLE_FONTSIZE)
            if row < len(logs) - 1:
                ax.tick_params(labelbottom=False)
        axes[row][0].set_ylabel(_short(label).replace(" ", "\n", 1), fontsize=TITLE_FONTSIZE,
                                rotation=0, ha="right", va="center", labelpad=22)

    fig.supxlabel("Batch size", fontsize=LABEL_FONTSIZE)
    fig.tight_layout()
    # the axis title goes left of the policy names, with a gap, which
    # tight_layout does not leave when it is a figure label
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    left = min(to_fig.transform(ax.yaxis.label.get_window_extent())[0][0] for ax in axes[:, 0])
    fig.supylabel("Share of the model's tasks", fontsize=LABEL_FONTSIZE,
                  x=left - 0.015, ha="right")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_drop_stages(jobs: list[pd.DataFrame], labels: list[str], colors: list, out: str):
    models = sorted({int(m) for j in jobs if "drop_model_id" in j
                     for m in j["drop_model_id"].dropna()} | set(MODEL_NAMES))
    stages = [MODEL_NAMES.get(m, f"model {m}") for m in models]

    # only the policies that drop jobs in flight: no drop sheds nothing, and
    # admission control rejects jobs before they reach any stage
    shown = [(j, l, c) for j, l, c in zip(jobs, labels, colors)
             if j["drop_model_id"].notna().any()]
    ncols = min(3, len(shown))
    nrows = -(-len(shown) // ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                             figsize=(5.2 * ncols, 3.6 * nrows + 1))
    peak = 0
    for ax, (job, label, color) in zip(axes.flat, shown):
        by_model = job["drop_model_id"].dropna().astype(int).value_counts()
        counts = [by_model.get(m, 0) for m in models]
        shares = np.array(counts) / len(job)
        peak = max(peak, shares.max())

        bars = ax.bar(range(len(stages)), shares, color=color, width=0.7)
        for bar, share in zip(bars, shares):
            if share > 0:
                ax.annotate(f"{100 * share:.1f}%", (bar.get_x() + bar.get_width() / 2, share),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=DROP_BAR_FONTSIZE, color="0.25")
        ax.set_title(f"{_short(label)}: {100 * shares.sum():.1f}% dropped",
                     loc="left", fontsize=DROP_TITLE_FONTSIZE)
        _style(ax, DROP_TICK_FONTSIZE)

    for ax in axes.flat[len(shown):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xticks(range(len(stages)), stages, rotation=35, ha="right")
    # leave room above the tallest bar for its label
    axes[0][0].set_ylim(0, peak * 1.2)
    axes[0][0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))

    fig.supxlabel("Stage the job was dropped at", fontsize=DROP_LABEL_FONTSIZE)
    fig.supylabel("Share of all jobs", fontsize=DROP_LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot(src: str, family: str, workload: str, stage_src: str | None):
    logs, labels = prt.policy_runs(src, family, workload)
    colors = list(sns.color_palette("tab10", len(logs)))
    fig_dir = os.path.join(src, "figures")
    name = f"{NAMES[family]}_{workload}"

    staged = [_stage_logs(d, stage_src) for d in logs]
    plot_batch_sizes(staged, labels, colors, os.path.join(fig_dir, f"{name}_batch_sizes.png"))
    plot_drop_stages([_stage_jobs(d) for d in staged], labels, colors,
                     os.path.join(fig_dir, f"{name}_drop_stages.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=str, required=True,
                        help="run_rrdrop_sweep.py results directory")
    parser.add_argument("--families", nargs="+", default=["wf13"],
                        help="Families to plot (default: wf13, the only one with "
                             "model names here)")
    parser.add_argument("--stage-src", type=str,
                        help="Rerun of the drop policies that logs drop_model_id, "
                             "for runs that predate it")
    args = parser.parse_args()

    for family, workload in CORE_DIRS:
        if family in args.families:
            plot(args.src, family, workload, args.stage_src)
