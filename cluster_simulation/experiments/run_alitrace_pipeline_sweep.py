"""Run an AliTrace policy matrix against a configured workflow.

Run from ``cluster_simulation/`` with:
    python experiments/run_alitrace_pipeline_sweep.py --name alitrace_wf6
    python experiments/run_alitrace_pipeline_sweep.py --workflow-id 12 \
        --allocation-family wf12 --name alitrace_wf12

The default workflow, trace, and allocation reproduce the WF6 AliTrace run. Token
bucket tuning runs use burst depths of 1, 5, 10, and 20 jobs, alongside the
automatically derived depth.
"""

import argparse
import json
import os
import subprocess
import sys

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = "workflow/alitrace/gpu_trace_2025/filtered_trace.csv"
WORKFLOW_ID = 6
ALLOCATION_FAMILY = "wf6"

# The canonical Vortex allocation used by run_core_sweep (task by task).
POLICIES = {
    "nodrop": {"DROP_POLICY": "NONE", "SLO_TYPE": "JOB_LEVEL",
               "ADMISSION_CONTROL_POLICY": "NONE"},
    "early": {"DROP_POLICY": "EARLY", "SLO_TYPE": "JOB_LEVEL",
              "ADMISSION_CONTROL_POLICY": "NONE"},
    "lookahead": {"DROP_POLICY": "LOOKAHEAD", "SLO_TYPE": "JOB_LEVEL",
                  "ADMISSION_CONTROL_POLICY": "NONE"},
    "nexus": {"DROP_POLICY": "EARLY", "SLO_TYPE": "NEXUS",
              "ADMISSION_CONTROL_POLICY": "NONE"},
    "flat10": {"DROP_POLICY": "NONE", "SLO_TYPE": "JOB_LEVEL",
               "ADMISSION_CONTROL_POLICY": "ROUND_ROBIN",
               "ADMISSION_DROP_RATE": 0.1},
    "token_auto": {"DROP_POLICY": "NONE", "SLO_TYPE": "JOB_LEVEL",
                   "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
                   "ADMISSION_BURST_SIZE": None},
}
for _burst_size in (1, 5, 10, 20):
    POLICIES[f"token_burst{_burst_size}"] = {
        "DROP_POLICY": "NONE", "SLO_TYPE": "JOB_LEVEL",
        "ADMISSION_CONTROL_POLICY": "TOKEN_BUCKET",
        "ADMISSION_BURST_SIZE": _burst_size,
    }


def _child_env():
    env = os.environ.copy()
    env["SIMULATION_DIR"] = SIM_DIR
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (SIM_DIR, env.get("PYTHONPATH", "")) if p)
    return env


def _resolve_allocation_family(workflow_id, requested_family):
    sys.path.insert(0, SIM_DIR)
    from experiments import run_core_sweep

    if requested_family:
        return requested_family

    canonical = f"wf{workflow_id}"
    if canonical in run_core_sweep.ALLOCATIONS:
        return canonical

    candidates = [family for family, workflow_ids in run_core_sweep.FAMILIES.items()
                  if workflow_id in workflow_ids]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"No Vortex allocation family includes workflow {workflow_id}")
    raise ValueError(f"Workflow {workflow_id} has multiple allocation families "
                     f"({', '.join(candidates)}); specify --allocation-family")


def _run_one(policy, output, workflow_id, trace, allocation_family):
    sys.path.insert(0, SIM_DIR)
    import core.configs.gen_config as gcfg
    from experiments import run_core_sweep
    from run_experiment import run_experiment

    if allocation_family not in run_core_sweep.ALLOCATIONS:
        raise ValueError(f"Unknown allocation family {allocation_family!r}; available: "
                         f"{', '.join(run_core_sweep.ALLOCATIONS)}")
    if workflow_id not in run_core_sweep.FAMILIES.get(allocation_family, []):
        raise ValueError(f"Workflow {workflow_id} is not part of allocation family "
                         f"{allocation_family!r}")
    allocation = run_core_sweep.ALLOCATIONS[allocation_family]
    runtime = run_core_sweep.job_runtime(allocation_family, workflow_id)
    slo_multiple = run_core_sweep.SLO_MULTIPLES.get(allocation_family, 5)
    overrides = {
        "CLIENT_CONFIGS": [{workflow_id: {
            "WORKLOAD": {"ARRIVAL_PROCESS": "ALITRACE", "TRACE_FILE_PATH": trace},
            "SLO": int(runtime * slo_multiple),
        }}],
        "ALLOCATION_STRATEGY": "CUSTOM",
        "CUSTOM_ALLOCATION": allocation,
        "DISPATCH_POLICY": "ROUND_ROBIN",
        "PRODUCE_EVENT_LOG": False,
        "EXEC_TIME_DISTRIBUTION": "NORMAL",
        "BOOST_POLICY": "FCFS",
    }
    overrides.update(POLICIES[policy])
    for option, value in overrides.items():
        setattr(gcfg, option, value)

    os.makedirs(os.path.join(output, "sim_logs"), exist_ok=True)
    run_experiment(True, output)
    with open(os.path.join(output, "overrides.json"), "w") as f:
        json.dump({"config": f"wf{workflow_id}_{policy}", "trace": trace,
                   "workflow_id": workflow_id,
                   "allocation_family": allocation_family,
                   "slo_multiple": slo_multiple,
                   "overrides": overrides}, f, indent=2)
    with open(os.path.join(output, "configs", "gen_config.py"), "a") as f:
        f.write(f"\n\n# Overrides applied by run_alitrace_pipeline_sweep.py ({policy})\n")
        for key, value in overrides.items():
            f.write(f"{key} = {value!r}\n")


def _run(policies, out_dir, parallel, workflow_id, trace, allocation_family):
    running, failed, pending = [], [], list(policies)
    while pending or running:
        while pending and len(running) < parallel:
            policy = pending.pop(0)
            name = f"wf{workflow_id}_{policy}"
            path = os.path.join(out_dir, name)
            os.makedirs(path, exist_ok=True)
            log = open(os.path.join(path, "run.log"), "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__),
                 "--run-one", policy, "--out", path,
                 "--workflow-id", str(workflow_id),
                 "--trace", trace,
                 "--allocation-family", allocation_family],
                # Repeat run configuration in each child process so every run
                # uses the same workflow, trace, and Vortex allocation.
                cwd=SIM_DIR, env=_child_env(), stdout=log,
                stderr=subprocess.STDOUT)
            running.append((name, proc, log))
            print(f"[start] {name} -> {path}", flush=True)
        for item in list(running):
            name, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                if proc.returncode:
                    failed.append(name)
                    print(f"[FAILED] {name} (exit {proc.returncode})", flush=True)
                else:
                    print(f"[done] {name}", flush=True)
        if running:
            import time
            time.sleep(1)
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None,
                        help="Output directory name (default: alitrace_wf<workflow-id>)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workflow-id", type=int, default=WORKFLOW_ID)
    parser.add_argument("--allocation-family", default=None,
                        help="Key in run_core_sweep.ALLOCATIONS; defaults to wf<id>")
    parser.add_argument("--trace", default=TRACE,
                        help="AliTrace CSV path, relative to the repository root or absolute")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--policies", nargs="+", choices=POLICIES.keys(),
                        default=list(POLICIES))
    parser.add_argument("--run-one", choices=POLICIES.keys(), default=None)
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    allocation_family = _resolve_allocation_family(
        args.workflow_id, args.allocation_family)
    if args.run_one:
        if not args.out:
            parser.error("--out is required with --run-one")
        _run_one(args.run_one, args.out, args.workflow_id, args.trace,
                 allocation_family)
        return
    name = args.name or f"alitrace_wf{args.workflow_id}"
    out_dir = args.out or os.path.join(SIM_DIR, "results", name)
    failed = _run(args.policies, out_dir, args.parallel, args.workflow_id,
                  args.trace, allocation_family)
    if failed:
        raise SystemExit(f"Failed runs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
