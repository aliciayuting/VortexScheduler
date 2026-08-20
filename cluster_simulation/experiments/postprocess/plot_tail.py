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

def plot_response_time_tail_cdf(srcs: list[tuple[str, str]], split_by_workflow: bool,
                                save_fig: bool, out_path: str):
    palette = sns.color_palette("tab10", len(srcs))

    max_res = 0
    loaded_srcs = []

    # Load and preprocess all sources once
    for dir, name in srcs:
        data = None
        if os.path.exists(os.path.join(dir, "job_breakdown.csv")):
            data = pd.read_csv(os.path.join(dir, "job_breakdown.csv"))
            data = data.rename(columns={"workflow_type": "workflow_id"})

            drop_path = os.path.join(dir, "drop_log.csv")
            if os.path.exists(drop_path):
                for _, dropped_row in pd.read_csv(drop_path).iterrows():
                    data.loc[len(data)] = {
                        "workflow_id": dropped_row["workflow_id"],
                        "job_create_time": dropped_row["create_time"],
                        "response_time": np.inf
                    }
        else:
            data = pd.read_csv(os.path.join(dir, "job_log.csv"))
            data.loc[data["was_completed"]==False, "response_time"] = np.inf

        finite_max = data.loc[np.isfinite(data["response_time"]), "response_time"]
        if len(finite_max) > 0:
            max_res = max(max_res, int(finite_max.max()) + 1)

        loaded_srcs.append((name, data))

    thresholds = np.arange(0, max_res, 1)

    if split_by_workflow:
        # collect all workflows across all loaded dataframes
        all_workflows = sorted(set().union(*[
            set(data["workflow_id"]) for name, data in loaded_srcs
        ]))
        n = len(all_workflows)

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))

        if n == 1:
            axes = [axes]

        wf2ax = {wf: ax for wf, ax in zip(all_workflows, axes)}

        for ax, wf in zip(axes, all_workflows):
            ax.set_title(f"Workflow {wf}")
            ax.set_yscale("log")
            ax.grid(True, which="both")

        for i, (name, data) in enumerate(loaded_srcs):
            workflows = sorted(set(data["workflow_id"]))

            for wf in workflows:
                ax = wf2ax[wf]

                subset = data.loc[data["workflow_id"] == wf, "response_time"].values
                cdf = np.array([(subset > t).mean() for t in thresholds])
                cdf[cdf == 0] = np.nan

                ax.plot(
                    thresholds,
                    cdf,
                    label=name,
                    color=palette[i],
                )

        for ax in axes:
            ax.legend()

        axes[-1].set_xlabel("Response time (ms)")
        plt.tight_layout()

    else:
        plt.figure(figsize=(8, 6))

        for i, (name, data) in enumerate(loaded_srcs):
            cdf = np.array([(data["response_time"] > t).mean() for t in thresholds])
            cdf[cdf == 0] = np.nan

            plt.plot(thresholds, cdf, label=name)

        plt.xlabel("Response time (ms)")
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


def plot_slo_as_job_size_vs_tail_cdf(srcs: list[tuple[str, str]], save_fig: bool, out_path: str):
    # TODO: For now we fix workflow execution times.
    palette = sns.color_palette("tab10", len(srcs))

    max_res = 0
    loaded_srcs = []

    # Load and preprocess all sources once
    for dir, name in srcs:
        data = None
        if os.path.exists(os.path.join(dir, "job_breakdown.csv")):
            data = pd.read_csv(os.path.join(dir, "job_breakdown.csv"))
            data = data.rename(columns={"workflow_type": "workflow_id"})

            drop_path = os.path.join(dir, "drop_log.csv")
            if os.path.exists(drop_path):
                for _, dropped_row in pd.read_csv(drop_path).iterrows():
                    data.loc[len(data)] = {
                        "workflow_id": dropped_row["workflow_id"],
                        "job_create_time": dropped_row["create_time"],
                        "response_time": np.inf
                    }
        else:
            data = pd.read_csv(os.path.join(dir, "job_log.csv"))
            data.loc[data["was_completed"]==False, "response_time"] = np.inf

        finite_max = data.loc[np.isfinite(data["response_time"]), "response_time"]
        if len(finite_max) > 0:
            max_res = max(max_res, int(finite_max.max()) + 1)

        loaded_srcs.append((name, data, get_workflow_job_sizes(dir)))

    thresholds = np.linspace(0, 8, 200)
    for i, (name, data, job_sizes) in enumerate(loaded_srcs):
        cts = None
        workflows = sorted(set(data['workflow_id']))
        for wf in workflows:
            subset = data.loc[data['workflow_id'] == wf, 'response_time'].values
            if cts is None:
                cts = np.array([(subset > t * job_sizes[wf]).sum() for t in thresholds])
            else:
                cts += np.array([(subset > t * job_sizes[wf]).sum() for t in thresholds])
            
        cdf = cts / len(data)
        cdf[cdf == 0] = np.nan
        plt.plot(
            thresholds,
            cdf,
            label=name,
            color=palette[i],
        )
    
    plt.xlabel('Response time as multiple of job size')
    plt.ylabel('Tail CDF (log)')
    plt.yscale('log')
    plt.grid(True, which='both')
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
    parser.add_argument("--slo-as-job-size", action="store_true",
                        help="Plot response time as a multiple of job size (the "
                             "workflow's critical path at batch size 1) instead of "
                             "in absolute ms")

    args = parser.parse_args()

    srcs = [(args.srcs[i], args.labels[i]) for i in range(len(args.srcs))]

    if args.slo_as_job_size:
        plot_slo_as_job_size_vs_tail_cdf(srcs, args.pdf, 
                                        args.out if args.out else "slo_as_job_size_tail.pdf")
    else:
        plot_response_time_tail_cdf(srcs, args.split, args.pdf, 
                                args.out if args.out else ("tail_by_workflow.pdf" if args.split else "tail_agg.pdf"))
