import os
import sys

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


def plot_batch_sizes(df: pd.DataFrame, is_batch_weighted: bool, split_by_model: bool, out_path: str | None):
    model_ids = sorted(list(set(df["model_id"])))

    nrows = math.ceil(len(model_ids) / 4) if split_by_model else 1
    ncols = 1 if (not split_by_model) or len(model_ids) == 1 else 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, model_id in zip(axes, model_ids):
        ax.grid(True, which="both")

        ax_df = df
        if split_by_model:
            ax_df = df[df["model_id"]==model_id]
            ax.set_title(f"Model {model_id}")
        
        if is_batch_weighted:
            ax.hist(ax_df["batch_size"], bins=np.arange(1, max(4, max(ax_df["batch_size"]) + 1)))
            ax.set_xlabel("Batch size")
            ax.set_ylabel("Batch count")
        else:
            ax.hist(ax_df["batch_size"], bins=np.arange(1, max(4, max(ax_df["batch_size"]) + 1)), 
                    weights=ax_df["batch_size"])
            ax.set_xlabel("Batch size")
            ax.set_ylabel("Job count")

    for i in range(nrows * ncols):
        if i >= len(model_ids):
            axes[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.25)
    fig.suptitle(f"Batch size histogram over {'batches' if is_batch_weighted else 'jobs'}")

    if out_path:
        plt.savefig(os.path.join(out_path, 
                                 f"bsize_hist_{'batches' if is_batch_weighted else 'jobs'}.pdf"))
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src", type=str, required=True, 
                        help="Root directory of simulation results")
    parser.add_argument("--pdf", action="store_true",
                        help="Save as PDF instead of launching plot")
    parser.add_argument("--split", action="store_true",
                        help="Split by model")
    parser.add_argument("--out", type=str, default="results",
                        help="Output directory path for saved figures")

    args = parser.parse_args()

    if args.pdf:
        os.makedirs(args.out, exist_ok=True)

    batch_df = pd.read_csv(os.path.join(args.src, "worker_batch_log.csv"))
    plot_batch_sizes(batch_df, True, args.split, args.out if args.pdf else None)
    plot_batch_sizes(batch_df, False, args.split, args.out if args.pdf else None)
    