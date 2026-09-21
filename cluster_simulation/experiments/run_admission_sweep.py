"""Runs the central round robin drop / admission control sweep.

Every configuration in SWEEP_CONFIGS is run against the workflow, allocation and
workload currently set in core/configs/gen_config.py; each run only overrides the
options listed for it, and results land in [out]/[config name]/.

Each configuration runs in its own process so that the RNG seeding in
run_experiment.py is identical across configurations, i.e. every configuration
sees the same arrival trace.

Usage (from cluster_simulation/, after `source set_env.sh`):
    python experiments/run_admission_sweep.py --name wf6
"""

import argparse
import json
import os
import subprocess
import sys

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(SIM_DIR, "experiments")

# every run is centralized round robin; only the drop / admission options vary
SWEEP_CONFIGS = {
    "central_rr_nodrop": {"DROP_POLICY": "NONE",
                          "ADMISSION_CONTROL_POLICY": "NONE"},
    "central_rr_lazy":   {"DROP_POLICY": "LAZY",
                          "SLO_TYPE": "JOB_LEVEL",
                          "ADMISSION_CONTROL_POLICY": "NONE"},
    "central_rr_early":  {"DROP_POLICY": "EARLY",
                          "SLO_TYPE": "JOB_LEVEL",
                          "ADMISSION_CONTROL_POLICY": "NONE"},
    "central_rr_nexus":  {"DROP_POLICY": "EARLY",
                          "SLO_TYPE": "NEXUS",
                          "ADMISSION_CONTROL_POLICY": "NONE"},
    # no dropping, but with the Nexus per-stage SLO split in place, recording
    # which jobs Nexus would have dropped instead of dropping them
    "central_rr_nexus_shadow": {"DROP_POLICY": "NONE",
                                "SLO_TYPE": "NEXUS",
                                "SHADOW_DROP_POLICY": "EARLY",
                                "ADMISSION_CONTROL_POLICY": "NONE"},
    "central_rr_10prob": {"DROP_POLICY": "NONE",
                          "ADMISSION_CONTROL_POLICY": "PROBABILISTIC",
                          "ADMISSION_DROP_RATE": 0.1},
    "central_rr_10rr":   {"DROP_POLICY": "NONE",
                          "ADMISSION_CONTROL_POLICY": "ROUND_ROBIN",
                          "ADMISSION_DROP_RATE": 0.1},
    "central_rr_token":  {"DROP_POLICY": "NONE",
                          "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
                          "ADMISSION_BURST_SIZE": None},
    "central_rr_5token": {"DROP_POLICY": "NONE",
                          "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
                          "ADMISSION_BURST_SIZE": 5},
    "central_rr_token_nexus": {"DROP_POLICY": "EARLY",
                               "SLO_TYPE": "NEXUS",
                               "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
                               "ADMISSION_BURST_SIZE": None},

    # the queue aware drop policy, which sheds a job once the work already in the
    # cluster means its remaining stages cannot finish in time. Paired against
    # central_rr_early, which asks the same question of an empty cluster
    "central_rr_lookahead": {"DROP_POLICY": "LOOKAHEAD",
                             "SLO_TYPE": "JOB_LEVEL",
                             "ADMISSION_CONTROL_POLICY": "NONE"},

    # the same two token bucket configurations metered per pipeline stage rather
    # than per job: a task is discarded before it runs when its stage's bucket is
    # empty, which ends its job and wastes whatever it already spent upstream
    "central_rr_token_task": {"DROP_POLICY": "NONE",
                              "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
                              "ADMISSION_GRANULARITY": "TASK",
                              "ADMISSION_BURST_SIZE": None},
    "central_rr_5token_task": {"DROP_POLICY": "NONE",
                               "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
                               "ADMISSION_GRANULARITY": "TASK",
                               "ADMISSION_BURST_SIZE": 5},

    # the three token bucket configurations repeated at a higher target
    # utilization, i.e. admitting a larger fraction of estimated capacity
    "central_rr_token_95": {"DROP_POLICY": "NONE",
                            "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
                            "ADMISSION_BURST_SIZE": None,
                            "ADMISSION_TARGET_UTILIZATION": 0.95},
    "central_rr_5token_95": {"DROP_POLICY": "NONE",
                             "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
                             "ADMISSION_BURST_SIZE": 5,
                             "ADMISSION_TARGET_UTILIZATION": 0.95},
    "central_rr_token_nexus_95": {"DROP_POLICY": "EARLY",
                                  "SLO_TYPE": "NEXUS",
                                  "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
                                  "ADMISSION_BURST_SIZE": None,
                                  "ADMISSION_TARGET_UTILIZATION": 0.95},
}


def _child_env() -> dict:
    env = os.environ.copy()
    env["SIMULATION_DIR"] = SIM_DIR
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [SIM_DIR, env.get("PYTHONPATH", "")] if p)
    return env


def run_one(config_name: str, out_path: str):
    """Runs a single sweep configuration into [out_path].

    Applies the configuration's overrides to the gen_config module, runs the
    simulation, then appends the overrides to the config copy saved alongside the
    results so that the saved config reflects what was actually run.
    """
    os.environ.setdefault("SIMULATION_DIR", SIM_DIR)
    for path in (SIM_DIR, EXP_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)

    import core.configs.gen_config as gcfg
    from run_experiment import run_experiment  # seeds the RNG on import

    overrides = dict(SWEEP_CONFIGS[config_name])
    overrides["DISPATCH_POLICY"] = "ROUND_ROBIN"
    for option, value in overrides.items():
        setattr(gcfg, option, value)

    os.makedirs(os.path.join(out_path, "sim_logs"), exist_ok=True)
    run_experiment(True, out_path)

    with open(os.path.join(out_path, "overrides.json"), "w") as f:
        json.dump({"config": config_name,
                   "scheduler_type": "central",
                   "overrides": overrides}, f, indent=2)

    # the config copy is written by run_experiment before the run, so append the
    # overrides (later assignments win) to keep it a faithful record
    with open(os.path.join(out_path, "configs", "gen_config.py"), "a") as f:
        f.write(f"\n\n\"\"\" ----- overrides applied by run_admission_sweep.py "
                f"({config_name}) ----- \"\"\"\n\n")
        for option, value in overrides.items():
            f.write(f"{option} = {value!r}\n")


def run_sweep(config_names: list[str], out_dir: str, max_parallel: int):
    """Runs each named configuration in its own process, up to [max_parallel] at once."""
    running: list[tuple[str, subprocess.Popen, object]] = []
    failed: list[str] = []
    pending = list(config_names)

    def reap(block: bool):
        for entry in list(running):
            name, proc, log = entry
            if not block and proc.poll() is None:
                continue
            code = proc.wait()
            log.close()
            running.remove(entry)
            if code == 0:
                print(f"[done] {name}")
            else:
                failed.append(name)
                print(f"[FAILED] {name} (exit {code}), see "
                      f"{os.path.join(out_dir, name, 'run.log')}")

    while pending or running:
        while pending and len(running) < max_parallel:
            name = pending.pop(0)
            run_path = os.path.join(out_dir, name)
            os.makedirs(run_path, exist_ok=True)
            log = open(os.path.join(run_path, "run.log"), "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__),
                 "--run-one", name, "--out", run_path],
                cwd=SIM_DIR, env=_child_env(), stdout=log, stderr=subprocess.STDOUT)
            running.append((name, proc, log))
            print(f"[start] {name} -> {run_path}")

        reap(block=len(running) >= max_parallel or not pending)

    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--name", type=str,
                        help="Name of the sweep, used as the results directory name")
    parser.add_argument("-o", "--out", type=str,
                        help="Results directory (default: experiments/results/[name])")
    parser.add_argument("-c", "--configs", nargs="+", choices=sorted(SWEEP_CONFIGS),
                        default=sorted(SWEEP_CONFIGS),
                        help="Subset of configurations to run (default: all)")
    parser.add_argument("-j", "--jobs", type=int, default=len(SWEEP_CONFIGS),
                        help="Number of configurations to run in parallel")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results instead of erroring out")
    parser.add_argument("--run-one", type=str, choices=sorted(SWEEP_CONFIGS),
                        help=argparse.SUPPRESS)  # internal: run a single configuration

    args = parser.parse_args()

    if args.run_one:
        run_one(args.run_one, args.out)
        sys.exit(0)

    if not args.name and not args.out:
        parser.error("one of --name or --out is required")

    out_dir = args.out or os.path.join(EXP_DIR, "results", args.name)
    out_dir = os.path.abspath(out_dir)

    existing = [name for name in args.configs
                if os.path.isdir(os.path.join(out_dir, name))
                and os.listdir(os.path.join(out_dir, name))]
    if existing and not args.overwrite:
        parser.error(f"results already exist in {out_dir} for: {', '.join(existing)}. "
                     "Pass --overwrite to replace them or pick another --name.")

    os.makedirs(out_dir, exist_ok=True)
    failed = run_sweep(args.configs, out_dir, max(1, args.jobs))

    print(f"\nResults in {out_dir}")
    if failed:
        print(f"{len(failed)} configuration(s) failed: {', '.join(failed)}")
        sys.exit(1)
