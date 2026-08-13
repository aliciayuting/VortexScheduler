import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

from scipy.stats import linregress

WORKFLOW_TYPE_TO_EXEC_TIME = {
    6: 62.5,
    7: 70.5,
    8: 80.5,
    1: 256.3,
    4: 787.2,
    5: 388.7
}

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

        loaded_srcs.append((name, data))

    thresholds = np.linspace(0, 8, 200)
    for i, (name, data) in enumerate(loaded_srcs):
        cts = None
        workflows = sorted(set(data['workflow_id']))
        for wf in workflows:
            subset = data.loc[data['workflow_id'] == wf, 'response_time'].values
            if cts is None:
                cts = np.array([(subset > t * WORKFLOW_TYPE_TO_EXEC_TIME[wf]).sum() for t in thresholds])
            else:
                cts += np.array([(subset > t * WORKFLOW_TYPE_TO_EXEC_TIME[wf]).sum() for t in thresholds])
            
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
                        help="Plot SLO as multiple of job size instead of absolute response time")

    args = parser.parse_args()

    srcs = [(args.srcs[i], args.labels[i]) for i in range(len(args.srcs))]

    if args.slo_as_job_size:
        plot_slo_as_job_size_vs_tail_cdf(srcs, args.pdf, 
                                        args.out if args.out else "slo_as_job_size_tail.pdf")
    else:
        plot_response_time_tail_cdf(srcs, args.split, args.pdf, 
                                args.out if args.out else ("tail_by_workflow.pdf" if args.split else "tail_agg.pdf"))
