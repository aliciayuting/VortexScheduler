"""Runs the token bucket refill rate sweep over the single tenant workloads.

TOKEN_BUCKET admission refills each workflow's bucket at ADMISSION_TARGET_UTILIZATION
times the throughput the cluster is estimated to sustain for it (see
AdmissionController._build_buckets). This sweep moves that fraction from 0.8 to
1.1, i.e. from well under the capacity estimate to past it, holding everything
else at the core sweep's `token` policy.

Only the refill rate changes. The derived bucket size is proportional to the refill
rate (the burst that drains within the SLO's slack at that rate), so left alone it
would grow and shrink with it. ADMISSION_BURST_SCALE cancels that, holding every
bucket at the size it has at the configured ADMISSION_TARGET_UTILIZATION. The runs
at that utilization therefore reproduce the core sweep's `token` runs, and every
rate of a given workload sees the same arrival trace. Results land in
[out]/[config name]/.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/run_refill_rate_sweep.py --name refill_rate_sweep
"""

import argparse
import itertools
import json
import os
import subprocess
import sys

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(SIM_DIR, "experiments")

for _path in (SIM_DIR, EXP_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.environ.setdefault("SIMULATION_DIR", SIM_DIR)

import run_core_sweep

# core sweep families running a single workflow: pipeline 2 (WF1), textvision
# (WF6), object defect inspection (WF12), dairy cow teat health (WF13) and
# time-series defect inspection (WF14)
FAMILIES = ["ppl2", "wf6", "wf12", "wf13", "wf14"]

# refill rates, as fractions of the estimated capacity (ADMISSION_TARGET_UTILIZATION)
RATES = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]


def config_names() -> list[str]:
    return [f"{family}_{workload}_rate{rate:g}" for family, workload, rate
            in itertools.product(FAMILIES, ["const", "burst"], RATES)]


def _overrides(config_name: str) -> dict:
    import core.configs.gen_config as gcfg

    family, workload, rate = config_name.split("_")
    rate = float(rate[len("rate"):])

    overrides = run_core_sweep._overrides(f"{family}_{workload}_token", "NORMAL")
    overrides["ADMISSION_BURST_SIZE"] = None
    overrides["ADMISSION_TARGET_UTILIZATION"] = rate
    # the derived bucket size scales with the refill rate; undo that, so the bucket
    # stays the size the core sweep's `token` runs give it
    overrides["ADMISSION_BURST_SCALE"] = gcfg.ADMISSION_TARGET_UTILIZATION / rate
    return overrides


def run_one(config_name: str, out_path: str):
    """Runs a single sweep configuration into [out_path]."""
    import core.configs.gen_config as gcfg
    from run_experiment import run_experiment  # seeds the RNG on import

    overrides = _overrides(config_name)
    for option, value in overrides.items():
        setattr(gcfg, option, value)

    os.makedirs(os.path.join(out_path, "sim_logs"), exist_ok=True)
    run_experiment(True, out_path)

    with open(os.path.join(out_path, "overrides.json"), "w") as f:
        json.dump({"config": config_name, "scheduler_type": "central",
                   "overrides": overrides}, f, indent=2)

    # the config copy is written before the run, so append the overrides to keep
    # it a faithful record; plot_tail reads CUSTOM_ALLOCATION back out of it
    with open(os.path.join(out_path, "configs", "gen_config.py"), "a") as f:
        f.write(f"\n\n\"\"\" ----- overrides applied by run_refill_rate_sweep.py "
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

        # wait for the oldest run, then refill
        name, proc, log = running.pop(0)
        code = proc.wait()
        log.close()
        if code == 0:
            print(f"[done] {name}", flush=True)
        else:
            failed.append(name)
            print(f"[FAILED] {name} (exit {code}), see "
                  f"{os.path.join(out_dir, name, 'run.log')}", flush=True)

    return failed


if __name__ == "__main__":
    all_configs = config_names()

    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--run-one", type=str, choices=all_configs,
                        help=argparse.SUPPRESS)  # internal: run a single configuration

    args = parser.parse_args()

    if args.run_one:
        run_one(args.run_one, args.out)
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
    failed = run_sweep(args.configs, out_dir, max(1, args.jobs))

    print(f"\nResults in {out_dir}")
    if failed:
        print(f"{len(failed)} configuration(s) failed: {', '.join(failed)}")
        sys.exit(1)
