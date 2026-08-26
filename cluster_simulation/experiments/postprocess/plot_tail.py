import os
import importlib.util

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


def get_workflow_job_sizes(src: str) -> dict[int, float]:
    """Returns each workflow's job size, i.e. the minimum time needed to run the
    pipeline: the critical path through the DAG at batch size 1 with no queueing.

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

    # batch size 1 is always given directly in the config, so there is no need to
    # rebuild ModelData's regression over the larger batch sizes
    exec_times = {i: m["MIG_BATCH_EXEC_TIMES"][24][1] for i, m in enumerate(mcfg.MODELS)}

    job_sizes = {}
    for cfg in wcfg.WORKFLOW_LIST:
        tasks = {t["TASK_INDEX"]: t for t in cfg["TASKS"]}

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


def plot_response_time_tail_cdf(srcs: list[tuple[str, str]], split_by_workflow: bool,
                                normalize_by_job_size: bool, save_fig: bool, out_path: str):
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
    """
    palette = sns.color_palette("tab10", len(srcs))
    loaded_srcs, max_res = _load_sources(srcs, need_job_sizes=normalize_by_job_size)

    if normalize_by_job_size:
        thresholds = np.linspace(0, 10, 250)
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

                wf2ax[wf].plot(thresholds, cdf, label=name, color=palette[i])

        for ax in axes:
            ax.legend()
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

            plt.plot(thresholds, cdf, label=name, color=palette[i])

        plt.xlabel(xlabel)
        plt.ylabel("Tail CDF")
        plt.title("Tail CDF")
        plt.grid(True, which="both")
        plt.yscale("log")
        plt.legend()
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

    plot_response_time_tail_cdf(srcs, args.split, args.normalize, args.pdf, out_path)
