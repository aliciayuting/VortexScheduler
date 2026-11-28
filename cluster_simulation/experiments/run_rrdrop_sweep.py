"""Reruns the core sweep's flat rate baseline (`rrdrop`, round robin admission at a
fixed drop rate) at the share of jobs the best policy now leaves unserved.

run_core_sweep.py sets the `rrdrop` drop rate to 1 - the highest SLO attainment
among the other core policies. The policies compared against have since grown to
include the tuned ones the core plots draw: the optimal static token bucket and
the adaptive token bucket. Here the drop rate is 1 - the highest SLO attainment
among EARLY, Nexus, LOOKAHEAD, the token bucket, the optimal static token bucket
and the adaptive token bucket, each scored on the core SLO with dropped and
rejected jobs counting as misses.

Each run is otherwise the core sweep's `rrdrop` configuration, so it sees the same
arrival trace. Results land in [out]/[family]_[workload]_rrdrop/.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/run_rrdrop_sweep.py --name rrdrop_min_unserved
"""

import argparse
import json
import os
import subprocess
import sys
import time

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(SIM_DIR, "experiments")
RESULTS = os.path.join(EXP_DIR, "results")

for _path in (SIM_DIR, EXP_DIR, os.path.join(EXP_DIR, "postprocess")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.environ.setdefault("SIMULATION_DIR", SIM_DIR)

import run_core_sweep

# core results directory, per family and workload
CORE_DIRS = {("ppl2", "const"): "wf1_wf13_core", ("ppl2", "burst"): "wf1_wf13_core",
             ("ppl2", "alitrace"): "wf1_wf6_alitrace_core",
             ("wf6", "const"): "wf6_core", ("wf6", "burst"): "wf6_core",
             ("wf6", "alitrace"): "wf1_wf6_alitrace_core",
             ("wf13", "const"): "wf1_wf13_core", ("wf13", "burst"): "wf1_wf13_core",
             ("wf15", "const"): "wf15_core", ("wf15", "burst"): "wf15_core",
             ("wf15", "staggered200"): "wf15_staggered_core",
             ("wf15", "staggered240"): "wf15_staggered_core"}

# adaptive token bucket results (--grow-only-if-rejecting), per workload
ADAPTIVE_DIRS = {"const": "adaptive_bucket_rejgrow", "burst": "adaptive_bucket_rejgrow",
                 "alitrace": "adaptive_bucket_rejgrow_alitrace",
                 "staggered200": "adaptive_bucket_rejgrow_wf15_staggered",
                 "staggered240": "adaptive_bucket_rejgrow_wf15_staggered"}

# family name as lookahead_single's runs give it
NAMES = {"ppl2": "wf1", "wf6": "wf6", "wf13": "wf13", "wf15": "wf15"}

# core policies compared against, by run name suffix in the core directory
CORE_POLICIES = ["early", "nexus", "lookahead", "token"]


def config_names() -> list[str]:
    return [f"{family}_{workload}_rrdrop" for family, workload in CORE_DIRS]


def _slo_met(run: str, slo: float | None = None) -> float:
    """Share of [run]'s jobs that finished within [slo] ms (default: each job's own
    deadline), counting jobs that never finished as misses."""
    jobs = pd_read(os.path.join(run, "sim_logs", "job_log.csv"))
    slo = jobs["deadline"] - jobs["create_time"] if slo is None else slo
    return (jobs["was_completed"].astype(bool) & (jobs["response_time"] <= slo)).mean()


def pd_read(path: str):
    import pandas as pd
    return pd.read_csv(path)


def candidate_runs(family: str, workload: str) -> dict[str, tuple[str, float | None]]:
    """Returns policy -> (run directory, SLO to score it on, or None for its own)."""
    import plot_core_tails as pct

    core = os.path.join(RESULTS, CORE_DIRS[(family, workload)])
    runs = {p: (os.path.join(core, f"{family}_{workload}_{p}"), None)
            for p in CORE_POLICIES}
    runs["optimal"] = (pct.optimal_run(family, workload), None)
    runs["adaptive"] = (os.path.join(RESULTS, ADAPTIVE_DIRS[workload],
                                     f"{family}_{workload}_adaptive"), None)

    for policy, (run, _) in runs.items():
        assert(os.path.isdir(os.path.join(run, "sim_logs"))), \
            f"Missing {policy} results for {family} {workload}: {run}"
    return runs


def min_unserved(family: str, workload: str) -> tuple[float, str, dict[str, float]]:
    """Returns the lowest share of jobs any candidate policy left unserved, the
    policy that reached it, and every candidate's SLO attainment."""
    attainments = {policy: float(_slo_met(run, slo))
                   for policy, (run, slo) in candidate_runs(family, workload).items()}
    best = max(attainments, key=attainments.get)
    return 1 - attainments[best], best, attainments


def run_one(config_name: str, out_path: str):
    """Runs a single configuration into [out_path]."""
    import core.configs.gen_config as gcfg
    from run_experiment import run_experiment  # seeds the RNG on import

    family, workload, _ = config_name.split("_")
    drop_rate, best, attainments = min_unserved(family, workload)

    core = os.path.join(RESULTS, CORE_DIRS[(family, workload)])
    overrides = run_core_sweep._overrides(config_name, "NORMAL", core)
    overrides["ADMISSION_DROP_RATE"] = drop_rate
    for option, value in overrides.items():
        setattr(gcfg, option, value)

    os.makedirs(os.path.join(out_path, "sim_logs"), exist_ok=True)
    run_experiment(True, out_path)

    with open(os.path.join(out_path, "overrides.json"), "w") as f:
        json.dump({"config": config_name, "scheduler_type": "central",
                   "overrides": overrides,
                   "drop_rate_from": {"policy": best, "slo_met": attainments}},
                  f, indent=2)

    # the config copy is written before the run, so append the overrides to keep
    # it a faithful record; plot_tail reads CUSTOM_ALLOCATION back out of it
    with open(os.path.join(out_path, "configs", "gen_config.py"), "a") as f:
        f.write(f"\n\n\"\"\" ----- overrides applied by run_rrdrop_sweep.py "
                f"({config_name}) ----- \"\"\"\n\n")
        for option, value in overrides.items():
            literal = f'"{value}"' if isinstance(value, str) else repr(value)
            f.write(f"{option} = {literal}\n")


def run_sweep(names: list[str], out_dir: str, max_parallel: int) -> list[str]:
    """Runs each named configuration in its own process, up to [max_parallel] at once."""
    running, failed, pending = [], [], list(names)

    while pending or running:
        while pending and len(running) < max_parallel:
            name = pending.pop(0)
            run_path = os.path.join(out_dir, name)
            os.makedirs(run_path, exist_ok=True)
            log = open(os.path.join(run_path, "run.log"), "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__), "--run-one", name,
                 "--out", run_path],
                cwd=SIM_DIR, env=run_core_sweep._child_env(), stdout=log,
                stderr=subprocess.STDOUT)
            running.append((name, proc, log))
            print(f"[start] {name} -> {run_path}", flush=True)

        finished = [entry for entry in running if entry[1].poll() is not None]
        while not finished:
            time.sleep(1)
            finished = [entry for entry in running if entry[1].poll() is not None]

        for entry in finished:
            running.remove(entry)
            name, proc, log = entry
            log.close()
            if proc.returncode == 0:
                print(f"[done] {name}", flush=True)
            else:
                failed.append(name)
                print(f"[FAILED] {name} (exit {proc.returncode}), see "
                      f"{os.path.join(out_dir, name, 'run.log')}", flush=True)

    return failed


if __name__ == "__main__":
    all_configs = config_names()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--name", type=str,
                        help="Name of the sweep, used as the results directory name")
    parser.add_argument("-o", "--out", type=str,
                        help="Results directory (default: experiments/results/[name])")
    parser.add_argument("-c", "--configs", nargs="+", choices=all_configs,
                        default=all_configs,
                        help="Subset of configurations to run (default: all)")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="Number of configurations to run in parallel")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print each configuration's drop rate and exit")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results instead of erroring out")
    parser.add_argument("--run-one", type=str, choices=all_configs,
                        help=argparse.SUPPRESS)  # internal: run a single configuration

    args = parser.parse_args()

    if args.run_one:
        run_one(args.run_one, args.out)
        sys.exit(0)

    if args.dry_run:
        for name in args.configs:
            family, workload, _ = name.split("_")
            rate, best, _ = min_unserved(family, workload)
            print(f"{name:24} drop rate {rate:.4f} (from {best})")
        sys.exit(0)

    if not args.name and not args.out:
        parser.error("one of --name or --out is required")

    out_dir = os.path.abspath(args.out or os.path.join(RESULTS, args.name))

    existing = [name for name in args.configs
                if os.path.isdir(os.path.join(out_dir, name))
                and os.listdir(os.path.join(out_dir, name))]
    if existing and not args.overwrite:
        parser.error(f"results already exist in {out_dir} for {len(existing)} configs. "
                     "Pass --overwrite to replace them or pick another --name.")

    os.makedirs(out_dir, exist_ok=True)
    failed = run_sweep(args.configs, out_dir, max(1, args.jobs))

    print(f"\nResults in {out_dir}")
    if failed:
        print(f"{len(failed)} configuration(s) failed: {', '.join(failed)}")
        sys.exit(1)
