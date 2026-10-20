import math
import os
import importlib.util

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import argparse

from scipy.stats import linregress


def _load_config(path: str):
    spec = importlib.util.spec_from_file_location(
        f"results_{path.replace(os.sep, '_')}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_config_dir(src: str) -> str:
    """Locates the config snapshot copied next to a run's logs. [src] may be either
    the results root or the sim_logs directory inside it.
    """
    for candidate in [os.path.join(src, "configs"),
                      os.path.join(src, os.pardir, "configs")]:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"No configs/ directory found for {src}; expected it in that directory or "
        "its parent")


def _get_model_instance_counts(cfg_dir: str) -> dict[int, int]:
    """Returns how many instances of each model the run placed, read from the
    allocation in its config snapshot.

    Returns an empty map when the run did not place models from a fixed list, in
    which case [get_workflow_job_sizes] has nothing to divide a stage's tasks
    between and charges each of them for one execution.

    Args:
        cfg_dir: Config snapshot directory for one run

    Returns:
        instance_counts: Model ID -> number of instances placed
    """
    gcfg = _load_config(os.path.join(cfg_dir, "gen_config.py"))

    if getattr(gcfg, "ALLOCATION_STRATEGY", None) != "CUSTOM":
        return {}

    counts: dict[int, int] = {}
    for _, model_ids in getattr(gcfg, "CUSTOM_ALLOCATION", []):
        for model_id in model_ids:
            counts[model_id] = counts.get(model_id, 0) + 1

    return counts


def get_workflow_job_sizes(src: str) -> dict[int, float]:
    """Returns each workflow's job size, i.e. how long one job takes with the
    cluster to itself: the longest path through the DAG at batch size 1, with each
    stage charged for the rounds its own job forces it into.

    A stage that a job fans out into more tasks than its model has instances cannot
    run them all at once, and a batch takes at most one task per job, so those
    tasks queue behind each other even on an idle cluster. Such a stage is charged
    ceil(tasks / instances) executions rather than one. Where a job has no more
    tasks on a model than there are instances -- every stage of the textvision and
    ppl2 workflows, and most others -- this is exactly the critical path.

    Read from the config snapshot in the results directory rather than hardcoded,
    so a plot always reflects the configuration that run actually used.

    Args:
        src: Results directory for one run

    Returns:
        job_sizes: Workflow ID -> job size (ms)
    """
    cfg_dir = _find_config_dir(src)
    wcfg = _load_config(os.path.join(cfg_dir, "workflow_config.py"))
    mcfg = _load_config(os.path.join(cfg_dir, "model_config.py"))
    instance_counts = _get_model_instance_counts(cfg_dir)

    # batch size 1 is always given directly in the config, so there is no need to
    # rebuild ModelData's regression over the larger batch sizes
    exec_times = {i: m["MIG_BATCH_EXEC_TIMES"][24][1] for i, m in enumerate(mcfg.MODELS)}

    job_sizes = {}
    for cfg in wcfg.WORKFLOW_LIST:
        tasks = {t["TASK_INDEX"]: t for t in cfg["TASKS"]}

        # DAG depth of each task; only tasks at the same depth contend for a model,
        # since a later depth cannot start until the earlier one is done
        depths: dict[int, int] = {}

        def depth(tid: int) -> int:
            if tid not in depths:
                depths[tid] = max([depth(p) + 1 for p in tasks[tid]["PREV_TASK_INDEX"]],
                                  default=0)
            return depths[tid]

        # executions a stage of this job must run back to back, because the job has
        # more tasks on that model at that depth than the allocation gives it instances
        task_counts: dict[tuple[int, int], int] = {}
        for tid, task in tasks.items():
            key = (task["MODEL_ID"], depth(tid))
            task_counts[key] = task_counts.get(key, 0) + 1

        rounds = {key: math.ceil(count / max(1, instance_counts.get(key[0], 1)))
                  for key, count in task_counts.items()}

        # walk the DAG in topological order accumulating the longest path. A task
        # cannot start until every predecessor is done, hence max over predecessors.
        finish_by: dict[int, float] = {}
        ready = [t for t in tasks.values() if not t["PREV_TASK_INDEX"]]
        while ready:
            next_ready = []
            for task in ready:
                tid = task["TASK_INDEX"]
                if tid in finish_by:
                    continue

                start = max([finish_by[p] for p in task["PREV_TASK_INDEX"]], default=0)
                finish_by[tid] = start + (exec_times[task["MODEL_ID"]]
                                          * rounds[(task["MODEL_ID"], depth(tid))]
                                          if task["MODEL_ID"] >= 0 else 0)

                next_ready.extend([
                    tasks[n] for n in task["NEXT_TASK_INDEX"]
                    if n not in finish_by
                    and all(p in finish_by for p in tasks[n]["PREV_TASK_INDEX"])])

            ready = next_ready

        assert(len(finish_by) == len(tasks))
        job_sizes[cfg["JOB_TYPE"]] = max(finish_by.values())

    return job_sizes

def _load_source(src: str, need_job_sizes: bool) -> tuple[pd.DataFrame, dict[int, float]]:
    """Loads one run's per job response times.

    Jobs that never completed, whether dropped mid-flight or rejected on arrival by
    admission control, are given an infinite response time so that they show up in
    the tail rather than silently vanishing from it.

    Args:
        src: Results directory for one run
        need_job_sizes: Whether to also read the run's per workflow job sizes,
        which requires the config snapshot to be present

    Returns:
        data: Per job response times, with a workflow_id column
        job_sizes: Workflow ID -> job size (ms), or None if not requested
    """
    if os.path.exists(os.path.join(src, "job_breakdown.csv")):
        data = pd.read_csv(os.path.join(src, "job_breakdown.csv"))
        data = data.rename(columns={"workflow_type": "workflow_id"})

        drop_path = os.path.join(src, "drop_log.csv")
        if os.path.exists(drop_path):
            for _, dropped_row in pd.read_csv(drop_path).iterrows():
                data.loc[len(data)] = {
                    "workflow_id": dropped_row["workflow_id"],
                    "job_create_time": dropped_row["create_time"],
                    "response_time": np.inf
                }
    else:
        data = pd.read_csv(os.path.join(src, "job_log.csv"))
        data.loc[data["was_completed"]==False, "response_time"] = np.inf

    return data, (get_workflow_job_sizes(src) if need_job_sizes else None)


def _load_sources(srcs: list[tuple[str, str]],
                  need_job_sizes: bool) -> tuple[list[tuple[str, pd.DataFrame, dict]], int]:
    """Loads every run to compare.

    Returns:
        loaded_srcs: (name, per job data, job sizes) for each run
        max_res: Largest finite response time across all runs (ms), which bounds
        the x axis when response times are plotted in absolute terms
    """
    loaded_srcs = []
    max_res = 0

    for src, name in srcs:
        data, job_sizes = _load_source(src, need_job_sizes)

        finite = data.loc[np.isfinite(data["response_time"]), "response_time"]
        if len(finite) > 0:
            max_res = max(max_res, int(finite.max()) + 1)

        loaded_srcs.append((name, data, job_sizes))

    return loaded_srcs, max_res


# hues the groups are drawn from, in the order groups are first seen. Each group
# gets a sequential ramp of one hue, so that a series' shade reads as its position
# within its group while its hue still says which group it belongs to
_GROUP_RAMPS = ["Blues", "Oranges", "Greens", "Purples", "Greys"]


def _series_colors(groups: list[str] | None, count: int) -> list:
    """Returns the colour for each series.

    Without [groups] every series is its own category and takes the next
    categorical hue, which is the right encoding when the runs being compared have
    no order to them. With [groups] the series of a group are shades of one hue,
    light to dark in the order given, which suits a group whose members ARE
    ordered, such as one run per SLO.

    Args:
        groups: Group name per series, or None for one colour per series
        count: Number of series

    Returns:
        colors: One matplotlib colour per series
    """
    if groups is None:
        return list(sns.color_palette("tab10", count))

    assert(len(groups) == count), \
        f"got {len(groups)} groups for {count} sources; pass one group per source"

    order = list(dict.fromkeys(groups))
    assert(len(order) <= len(_GROUP_RAMPS)), \
        f"only {len(_GROUP_RAMPS)} group ramps are defined, got {len(order)} groups"

    colors = [None] * count
    for position, name in enumerate(order):
        members = [i for i, g in enumerate(groups) if g == name]
        ramp = plt.get_cmap(_GROUP_RAMPS[position])

        # stay inside the dark half of the ramp: its pale end is too close to the
        # page to read as a line, and with one member there is no ramp to walk
        steps = (np.linspace(0.40, 0.95, len(members)) if len(members) > 1
                 else np.array([0.75]))
        for step, i in zip(steps, members):
            colors[i] = ramp(step)

    return colors


# dash patterns the --styles keys take, in the order keys are first seen
_LINESTYLES = ["-", "--", ":", "-."]


def _keyed(keys: list[str], values: list, count: int, what: str) -> list:
    """Maps each series' key to a value, giving keys values in the order the keys
    are first seen, so that series sharing a key share the value.
    """
    assert(len(keys) == count), \
        f"got {len(keys)} {what} for {count} sources; pass one per source"
    order = list(dict.fromkeys(keys))
    assert(len(order) <= len(values)), \
        f"only {len(values)} {what} are available, got {len(order)} distinct ones"
    return [values[order.index(k)] for k in keys]


def _legend(ax, labels: list[str], colors: list, hues: list[str] | None,
            styles: list[str] | None, linestyles: list[str]):
    """Draws the legend. With [hues] or [styles] it lists each colour and each
    dash pattern once, rather than every series, so a comparison of N policies
    under M variants takes N + M entries instead of N * M.
    """
    if hues is None and styles is None:
        ax.legend(ncol=2 if len(labels) > 8 else 1, fontsize="small")
        return

    handles = []
    for key in dict.fromkeys(hues or labels):
        i = (hues or labels).index(key)
        handles.append(Line2D([], [], color=colors[i], linestyle="-", label=key))
    for key in dict.fromkeys(styles or []):
        i = styles.index(key)
        handles.append(Line2D([], [], color="0.3", linestyle=linestyles[i], label=key))
    ax.legend(handles=handles, fontsize="small")


def plot_response_time_tail_cdf(srcs: list[tuple[str, str]], split_by_workflow: bool,
                                normalize_by_job_size: bool, save_fig: bool, out_path: str,
                                max_job_size_multiple: float = 10,
                                groups: list[str] | None = None,
                                hues: list[str] | None = None,
                                styles: list[str] | None = None):
    """Plots the tail CDF of job response time for every run in [srcs].

    The two options are independent and may be combined:

    [normalize_by_job_size] measures each job's response time in multiples of its
    workflow's job size (the critical path through the DAG at batch size 1) rather
    than in ms, putting workflows with different pipeline lengths on a common x
    axis. [split_by_workflow] gives each workflow its own subplot instead of
    pooling every job into a single curve.

    Args:
        srcs: (results directory, label) for each run to compare
        split_by_workflow: One subplot per workflow rather than one pooled curve
        normalize_by_job_size: X axis in multiples of job size rather than ms
        save_fig: Write the figure to [out_path] instead of displaying it
        out_path: Where to save the figure
        max_job_size_multiple: Right edge of the x axis when normalizing, so that
        runs whose SLO exceeds the default still show their tail
        groups: Group name per run, which colours the runs of a group as shades of
        one hue instead of giving each its own (see [_series_colors])
        hues: Colour key per run. Runs sharing a key share a categorical colour,
        and the legend lists each key once. Overrides [groups]
        styles: Dash pattern key per run. Runs sharing a key share a dash pattern,
        and the legend lists each key once
    """
    labels = [name for _, name in srcs]
    if hues is not None:
        palette = _keyed(hues, list(sns.color_palette("tab10")), len(srcs), "hues")
    else:
        palette = _series_colors(groups, len(srcs))
    linestyles = (_keyed(styles, _LINESTYLES, len(srcs), "styles")
                  if styles is not None else ["-"] * len(srcs))
    loaded_srcs, max_res = _load_sources(srcs, need_job_sizes=normalize_by_job_size)

    if normalize_by_job_size:
        thresholds = np.linspace(0, max_job_size_multiple, 250)
        xlabel = "Response time as multiple of job size"
    else:
        thresholds = np.arange(0, max_res, 1)
        xlabel = "Response time (ms)"

    def _exceedance_counts(response_times, scale: float) -> np.ndarray:
        """Number of jobs whose response time exceeds each threshold. [scale] is
        the workflow's job size when normalizing, and 1 when plotting raw ms.
        """
        return np.array([(response_times > t * scale).sum() for t in thresholds])

    def _workflow_scale(job_sizes: dict, wf) -> float:
        return job_sizes[wf] if normalize_by_job_size else 1.0

    if split_by_workflow:
        # collect all workflows across all loaded dataframes
        all_workflows = sorted(set().union(*[
            set(data["workflow_id"]) for _, data, _ in loaded_srcs
        ]))

        fig, axes = plt.subplots(1, len(all_workflows),
                                 figsize=(4 * len(all_workflows), 6), squeeze=False)
        axes = axes[0]
        wf2ax = {wf: ax for wf, ax in zip(all_workflows, axes)}

        for ax, wf in zip(axes, all_workflows):
            ax.set_title(f"Workflow {wf}")
            ax.set_yscale("log")
            ax.grid(True, which="both")

        for i, (name, data, job_sizes) in enumerate(loaded_srcs):
            for wf in sorted(set(data["workflow_id"])):
                subset = data.loc[data["workflow_id"] == wf, "response_time"].values

                cdf = _exceedance_counts(subset, _workflow_scale(job_sizes, wf)) / len(subset)
                cdf[cdf == 0] = np.nan

                wf2ax[wf].plot(thresholds, cdf, label=name, color=palette[i],
                               linestyle=linestyles[i])

        for ax in axes:
            _legend(ax, labels, palette, hues, styles, linestyles)
            ax.set_xlabel(xlabel)

        axes[0].set_ylabel("Tail CDF")
        plt.tight_layout()

    else:
        plt.figure(figsize=(8, 6))

        for i, (name, data, job_sizes) in enumerate(loaded_srcs):
            # every workflow is measured against its own scale, then pooled into
            # one curve over all of the run's jobs
            counts = np.zeros(len(thresholds))
            for wf in sorted(set(data["workflow_id"])):
                subset = data.loc[data["workflow_id"] == wf, "response_time"].values
                counts += _exceedance_counts(subset, _workflow_scale(job_sizes, wf))

            cdf = counts / len(data)
            cdf[cdf == 0] = np.nan

            plt.plot(thresholds, cdf, label=name, color=palette[i],
                     linestyle=linestyles[i])

        plt.xlabel(xlabel)
        plt.ylabel("Tail CDF")
        plt.title("Tail CDF")
        plt.grid(True, which="both")
        plt.yscale("log")
        _legend(plt.gca(), labels, palette, hues, styles, linestyles)
        plt.tight_layout()

    if save_fig:
        plt.savefig(out_path)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--srcs", type=str, nargs="+", required=True, 
                        help="Root directories of simulation results to compare")
    parser.add_argument("--labels", type=str, nargs="+", required=True, 
                        help="Names to give each simulation run")
    parser.add_argument("--pdf", action="store_true",
                        help="Save as PDF instead of launching plot")
    parser.add_argument("--split", action="store_true",
                        help="Split by workflow")
    parser.add_argument("--out", type=str,
                        help="Output directory path for saved figures")
    parser.add_argument("--xmax", type=float, default=10,
                        help="With --normalize, the right edge of the x axis in "
                             "multiples of job size (default: 10)")
    parser.add_argument("--groups", type=str, nargs="+",
                        help="Group name per source. Sources sharing a group are "
                             "drawn as shades of one hue, light to dark in the "
                             "order given, rather than each taking its own hue")
    parser.add_argument("--hues", type=str, nargs="+",
                        help="Colour key per source. Sources sharing a key share a "
                             "colour and are listed once in the legend")
    parser.add_argument("--styles", type=str, nargs="+",
                        help="Dash pattern key per source. Sources sharing a key "
                             "share a dash pattern and are listed once in the legend")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize response time to a multiple of job size (the "
                             "workflow's critical path at batch size 1) instead of "
                             "plotting it in absolute ms. May be combined with --split")

    args = parser.parse_args()

    srcs = [(args.srcs[i], args.labels[i]) for i in range(len(args.srcs))]

    if args.out:
        out_path = args.out
    elif args.normalize:
        out_path = ("slo_as_job_size_tail_by_workflow.pdf" if args.split
                    else "slo_as_job_size_tail.pdf")
    else:
        out_path = "tail_by_workflow.pdf" if args.split else "tail_agg.pdf"

    plot_response_time_tail_cdf(srcs, args.split, args.normalize, args.pdf, out_path,
                                args.xmax, args.groups, args.hues, args.styles)
