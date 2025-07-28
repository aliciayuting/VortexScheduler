import os
import sys
import argparse
import json

from bidict import bidict

from simulations.simulation_central import *
from simulations.simulation_decentral import *


sys.dont_write_bytecode = True
np.random.seed(42)

# Scheduler options
NO_SCHEDULER = 0
DECENTRALHEFT = 1
CENTRALHEFT = 2
HASHTASK = 3
SHEPHERD = 4

SCHEDULER_NAMES = bidict({
    NO_SCHEDULER: "no_scheduler",
    DECENTRALHEFT: "decentralheft",
    CENTRALHEFT: "centralheft",
    HASHTASK: "hashtask",
    SHEPHERD: "shepherd"
})

def run_experiment(scheduler_type: int, job_types: list[int], out_path_root: str):
    assert(scheduler_type in [NO_SCHEDULER, DECENTRALHEFT, CENTRALHEFT, HASHTASK, SHEPHERD])
    assert(scheduler_type != SHEPHERD or not ENABLE_MULTITHREADING) # concurrency not implemented for Shepherd
    assert(ALLOCATION_STRATEGY != "HERD" or ENABLE_DYNAMIC_MODEL_LOADING) # if HERD, dynamic loading must be enabled

    out_path = os.path.join(out_path_root, SCHEDULER_NAMES[scheduler_type])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Created directory at {out_path}")
    else:
        print(f"Directory at {out_path} exists")

    sim = None
    if scheduler_type == CENTRALHEFT:
        sim = Simulation_central(simulation_name="centralheft", job_split="PER_TASK",
                                 num_workers=TOTAL_NUM_OF_NODES, job_types_list=job_types,
                                 produce_breakdown=True)
    elif scheduler_type == DECENTRALHEFT:
        sim = Simulation_decentral(simulation_name="decentralheft", job_split="PER_TASK",
                                   num_workers=TOTAL_NUM_OF_NODES, job_types_list=job_types,
                                   dynamic_adjust=False, consider_load=True, consider_cache=True, produce_breakdown=True)
    elif scheduler_type == HASHTASK:
        sim = Simulation_central(simulation_name="hashtask", job_split="PER_TASK",
                                 num_workers=TOTAL_NUM_OF_NODES, job_types_list=job_types,
                                 produce_breakdown=True)
    elif scheduler_type == SHEPHERD:
        sim = Simulation_central(simulation_name="shepherd", job_split="PER_TASK",
                                 num_workers=TOTAL_NUM_OF_NODES, job_types_list=job_types,
                                 produce_breakdown=True)

    sim.run()

    event_log = sim.event_log
    event_log.to_csv(os.path.join(out_path, "events_by_time.csv"))
    
    result_to_export = sim.result_to_export
    result_to_export.to_csv(os.path.join(out_path, "job_breakdown.csv"))

    tasks_logging_times = sim.tasks_logging_times
    tasks_logging_times.to_csv(os.path.join(out_path, "loadDelay_" + str(
        LOAD_INFORMATION_STALENESS) + "_placementDelay_" + str(PLACEMENT_INFORMATION_STALENESS) + ".csv"))
    
    sim.batch_exec_log.to_csv(os.path.join(out_path, "batch_log.csv"))
    
    worker_model_histories = pd.concat(list(map(lambda w: w.model_history_log, sim.workers)), 
                                    keys=list(map(lambda w: w.worker_id, sim.workers)), 
                                    names=['worker_id']).reset_index(level='worker_id')
    worker_model_histories = worker_model_histories.sort_values(by="start_time")
    worker_model_histories.to_csv(os.path.join(out_path, "model_history_log.csv"))

    sim.task_drop_log.to_csv(os.path.join(out_path, "drop_log.csv"))

    with open(os.path.join(out_path, "stats.json"), "w") as f:
        f.write(json.dumps(sim.sim_stats_log))
        f.close()

    print("** Simulation stats saved to stats.json **")

    if ALLOCATION_STRATEGY == "HERD":
        os.makedirs(os.path.join(out_path, "allocations"), exist_ok=True)
        for time, alog in sim.allocation_logs:
            with open(os.path.join(out_path, "allocations", f"herd_allocation_{time}.json"), "w") as f:
                f.write(alog)
                f.close()


if __name__ == "__main__":
    produce_breakdown = True
    job_types = [jt for c in CLIENT_CONFIGS for jt, v in c.items() if v["NUM_JOBS"] > 0]

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--scheduler-type", type=str, required=True, choices=["centralheft", "decentralheft", "shepherd", "hashtask"])
    parser.add_argument("-o", "--out", type=str, default="results", help="Path to output directory")
    
    args = parser.parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    scheduler_type = SCHEDULER_NAMES.inv[args.scheduler_type]
    run_experiment(scheduler_type, job_types, args.out)