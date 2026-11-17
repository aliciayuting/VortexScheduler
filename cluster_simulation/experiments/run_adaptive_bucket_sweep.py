"""Runs the adaptive token bucket prototype over the single tenant workloads.

Each run is the core sweep's `token` policy with two changes:

  - the refill rate is set to that of the best static token bucket found for that
    workflow and workload ([optimal_bucket], the same one the core plots draw as
    "token-bucket (optimal)"), and
  - the bucket size is adapted online (ADAPTIVE_BUCKET, see
    AdaptiveBucketSizer in schedulers/algo/admission_algo.py), starting from the
    default size: the one derived at the configured default
    ADMISSION_TARGET_UTILIZATION, which is also the bucket every run of the refill
    rate sweep and the 1x runs of the bucket size sweep used.

Every window of completed jobs, the sizer grows the bucket by one job if the
window's p99 response time is under (1 - margin) x SLO, halves it if the p99 is
over the SLO, and otherwise holds it, never leaving [3, 2.5 x default]. Two
static baselines are worth comparing against: the 1x run at the same rate in
the bucket size sweep that rate was taken from, which differs from these
runs only in leaving the bucket fixed at its default size, and the best static
token bucket itself, which also fixes the bucket at its best size. Results land
in [out]/[config name]/.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/run_adaptive_bucket_sweep.py --name adaptive_bucket
    python experiments/run_adaptive_bucket_sweep.py --name adaptive_bucket_t5s \\
        --window-type TIME --window 5000
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(SIM_DIR, "experiments")

for _path in (SIM_DIR, EXP_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.environ.setdefault("SIMULATION_DIR", SIM_DIR)

import run_core_sweep

# core sweep families running a single workflow: pipeline 2 (WF1), textvision
# (WF6), dairy cow teat health (WF13) and time-series defect inspection at a
# 600 ms round interval (WF15)
FAMILIES = ["ppl2", "wf6", "wf13", "wf15"]

def optimal_bucket(family: str, workload: str):
    """The best static token bucket for [family] and [workload], as
    postprocess/plot_core_tails.py picks it for its "token-bucket (optimal)" line:
    the bucket size sweep run with the highest SLO attainment across
    results/bucket_size_sweep_{optrate,altrate,rate1}. Read at run time, so the
    adaptive runs follow those sweeps if they are rerun.

    Returns:
        best: its summary.csv row; [refill_rate] is the fraction of estimated
        capacity, [sweep] the sweep it came from
    """
    sys.path.insert(0, os.path.join(EXP_DIR, "postprocess"))
    from plot_core_tails import optimal_bucket as pick
    return pick(family, workload)


def config_names() -> list[str]:
    return [f"{family}_{workload}_adaptive"
            for family, workload in itertools.product(FAMILIES, ["const", "burst"])]


def _overrides(config_name: str, tuning: dict) -> dict:
    import core.configs.gen_config as gcfg

    family, workload, _ = config_name.split("_")
    rate = float(optimal_bucket(family, workload)["refill_rate"])

    overrides = run_core_sweep._overrides(f"{family}_{workload}_token", "NORMAL")
    overrides["ADMISSION_BURST_SIZE"] = None
    overrides["ADMISSION_TARGET_UTILIZATION"] = rate
    # the derived bucket scales with the refill rate; undo that, so the bucket
    # starts at its default size, the one derived at the default rate
    overrides["ADMISSION_BURST_SCALE"] = gcfg.ADMISSION_TARGET_UTILIZATION / rate
    overrides["ADAPTIVE_BUCKET"] = True
    overrides.update(tuning)
    return overrides


def _tuning(args) -> dict:
    """The ADAPTIVE_BUCKET options set on the command line, as config overrides."""
    options = {"ADAPTIVE_BUCKET_WINDOW_TYPE": args.window_type,
               "ADAPTIVE_BUCKET_WINDOW": args.window,
               "ADAPTIVE_BUCKET_MIN_SAMPLES": args.min_samples,
               "ADAPTIVE_BUCKET_SLACK_MARGIN": args.margin,
               "ADAPTIVE_BUCKET_IGNORE_STALE": None if args.keep_stale is None
                   else not args.keep_stale,
               "ADAPTIVE_BUCKET_GROW_ONLY_IF_REJECTING":
                   True if args.grow_only_if_rejecting else None}
    return {k: v for k, v in options.items() if v is not None}


def _tuning_argv(tuning: dict) -> list[str]:
    flags = {"ADAPTIVE_BUCKET_WINDOW_TYPE": "--window-type",
             "ADAPTIVE_BUCKET_WINDOW": "--window",
             "ADAPTIVE_BUCKET_MIN_SAMPLES": "--min-samples",
             "ADAPTIVE_BUCKET_SLACK_MARGIN": "--margin"}
    argv = []
    for option, value in tuning.items():
        if option == "ADAPTIVE_BUCKET_IGNORE_STALE":
            argv += [] if value else ["--keep-stale"]
        elif option == "ADAPTIVE_BUCKET_GROW_ONLY_IF_REJECTING":
            argv += ["--grow-only-if-rejecting"] if value else []
        else:
            argv += [flags[option], str(value)]
    return argv


def run_one(config_name: str, out_path: str, tuning: dict):
    """Runs a single sweep configuration into [out_path]."""
    import core.configs.gen_config as gcfg
    from run_experiment import run_experiment  # seeds the RNG on import

    overrides = _overrides(config_name, tuning)
    for option, value in overrides.items():
        setattr(gcfg, option, value)

    os.makedirs(os.path.join(out_path, "sim_logs"), exist_ok=True)
    run_experiment(True, out_path)

    # which static bucket the refill rate was taken from, since it is picked at
    # run time and may change if the bucket size sweeps are rerun
    family, workload, _ = config_name.split("_")
    best = optimal_bucket(family, workload)
    with open(os.path.join(out_path, "overrides.json"), "w") as f:
        json.dump({"config": config_name, "scheduler_type": "central",
                   "overrides": overrides,
                   "refill_rate_from": {"sweep": os.path.basename(best["sweep"]),
                                        "scale": float(best["scale"]),
                                        "refill_rate": float(best["refill_rate"]),
                                        "slo_met": float(best["slo_met"])}},
                  f, indent=2)

    # the config copy is written before the run, so append the overrides to keep
    # it a faithful record; plot_tail reads CUSTOM_ALLOCATION back out of it
    with open(os.path.join(out_path, "configs", "gen_config.py"), "a") as f:
        f.write(f"\n\n\"\"\" ----- overrides applied by run_adaptive_bucket_sweep.py "
                f"({config_name}) ----- \"\"\"\n\n")
        for option, value in overrides.items():
            literal = f'"{value}"' if isinstance(value, str) else repr(value)
            f.write(f"{option} = {literal}\n")


def run_sweep(names: list[str], out_dir: str, max_parallel: int,
              tuning: dict) -> list[str]:
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
                 "--out", run_path] + _tuning_argv(tuning),
                cwd=SIM_DIR, env=run_core_sweep._child_env(), stdout=log,
                stderr=subprocess.STDOUT)
            running.append((name, proc, log))
            print(f"[start] {name} -> {run_path}", flush=True)

        # wait for any run to finish, then refill, so one slow run does not hold
        # up the slots the others free
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
    parser.add_argument("-j", "--jobs", type=int, default=8,
                        help="Number of configurations to run in parallel")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results instead of erroring out")

    tuning = parser.add_argument_group("adaptive bucket options (default: gen_config)")
    tuning.add_argument("--window-type", choices=["COUNT", "TIME"],
                        help="Decide every --window completed jobs (COUNT) or ms (TIME)")
    tuning.add_argument("--window", type=float,
                        help="Window length, in completed jobs or ms")
    tuning.add_argument("--min-samples", type=int,
                        help="TIME windows: fewest completions a decision is taken on")
    tuning.add_argument("--margin", type=float,
                        help="Grow only when p99 < (1 - margin) x SLO")
    tuning.add_argument("--keep-stale", action="store_true", default=None,
                        help="Also judge jobs admitted before the last resize")
    tuning.add_argument("--grow-only-if-rejecting", action="store_true",
                        help="Grow only if the bucket rejected a job since the last "
                             "decision")

    parser.add_argument("--run-one", type=str, choices=all_configs,
                        help=argparse.SUPPRESS)  # internal: run a single configuration

    args = parser.parse_args()
    if args.window is not None and args.window == int(args.window):
        args.window = int(args.window)

    if args.run_one:
        run_one(args.run_one, args.out, _tuning(args))
        sys.exit(0)

    if not args.name and not args.out:
        parser.error("one of --name or --out is required")

    out_dir = os.path.abspath(args.out or os.path.join(EXP_DIR, "results", args.name))

    existing = [name for name in args.configs
                if os.path.isdir(os.path.join(out_dir, name))
                and os.listdir(os.path.join(out_dir, name))]
    if existing and not args.overwrite:
        parser.error(f"results already exist in {out_dir} for {len(existing)} configs. "
                     "Pass --overwrite to replace them or pick another --name.")

    os.makedirs(out_dir, exist_ok=True)
    failed = run_sweep(args.configs, out_dir, max(1, args.jobs), _tuning(args))

    print(f"\nResults in {out_dir}")
    if failed:
        print(f"{len(failed)} configuration(s) failed: {', '.join(failed)}")
        sys.exit(1)
