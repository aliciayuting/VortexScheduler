'''
This simulation experiment framework of event generation 
is referenced from Sparrow: https://github.com/radlab/sparrow 
'''

import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

from queue import PriorityQueue

from core.config import *
from core.metadata_service import *
from core.print_utils import *
from core.external_client import *
from core.events import *

from workers.heft_task_worker import *
from workers.shepherd_task_worker import *

from schedulers.algo.herd_algo import get_herd_assignment
from schedulers.centralized.shepherd.herd_assignment import HerdAssignment


class Simulation(object):
    def __init__(
            self,
            simulation_name,
            job_split,  
            centralized_scheduler=False,  
            dynamic_adjust=False,  
            total_workers=1,
            slots_per_worker=1,
            job_types_list=[0],
            produce_breakdown=False
    ):
        self.simulation_name = simulation_name
        self.centralized_scheduler = centralized_scheduler
        self.job_split = job_split
        self.total_workers = total_workers
        self.slots_per_worker = slots_per_worker
        self.job_types_list = job_types_list
        self.job_split = job_split
        self.dynamic_adjust = dynamic_adjust
        self.workers = []
        self.metadata_service = MetadataService()
        self.external_clients = []

        self.remaining_jobs = sum(cc[jt]["NUM_JOBS"] for cc in CLIENT_CONFIGS for jt in cc.keys())
        self.event_queue = PriorityQueue()

        self.jobs = {}
        
        # Tracking measurements
        self.result_to_export = pd.DataFrame()
        self.tasks_logging_times = pd.DataFrame()
        self.event_log = pd.DataFrame(columns=["time", "worker_id", "event"])
        self.batch_exec_log = pd.DataFrame(columns=["start_time", "end_time", "worker_id", "workflow_id", 
                                                    "task_id", "batch_size", "job_ids"])
        self.task_drop_log = pd.DataFrame(columns=["job_id", "workflow_id", "task_id", "drop_time", 
                                                   "arrival_time", "slo", "deadline"])
        self.allocation_logs = []

        print("---- SIMULATION : " + self.simulation_name + "----")
        self.produce_breakdown =  produce_breakdown

    def get_model_from_id(self, model_id: int) -> Model:
        all_models = [m for ms in list(self.metadata_service.job_type_models.values()) for m in ms]
        return list(filter(lambda m: m.model_id == model_id, all_models))[0]
    
    def run_herd_scheduler(self, current_time: float):
        """
            Initializes a new HerdAssignment and stores in [self.herd_assignment].
        """
        curr_send_rates = {}
        for jt in self.job_types_list:
            jobs_since_last_sched = [j for j in self.jobs.values() if j.job_type_id == jt and \
                                        j.tasks[0].log.task_arrival_at_scheduler_timestamp > (current_time - HERD_PERIODICITY) and \
                                        j.tasks[0].log.task_arrival_at_scheduler_timestamp <= current_time]
            curr_send_rates[jt] = 5 if current_time == 0 else (len(jobs_since_last_sched) / HERD_PERIODICITY * 1000 + 5) 
            # TODO: minimum send rate? 
        self.workers = []

        task_types = get_task_types(self.job_types_list)
        models_by_wf = list(self.metadata_service.job_type_models.values())
        all_models = [m for jt in self.job_types_list for m in models_by_wf[jt]]
        task_tputs = {(0,0): 270, (0,1): 45, (0,2): 270, (0,3): 70,
                        (1,0): 125, (1,1): 7555, (1,2): 92, (1,3): 4.82}
        (group_sizes, stream_groups) = get_herd_assignment(
            task_types, all_models, task_tputs, curr_send_rates)
        
        worker_groups = []
        worker_counter = 0
        for i, group_size in enumerate(group_sizes):
            group_workers = []
            if self.simulation_name == "shepherd":
                group_workers = [ShepherdWorker(self, worker_counter+j, 24, i) for j in range(int(group_size))]
            else:
                group_workers = [HeftTaskWorker(self, worker_counter+j, 24, group_id=i) for j in range(int(group_size))]

            worker_counter += len(group_workers)
            self.workers += group_workers
            worker_groups.append(group_workers)

        task_type_assignments = {}
        for (sid, group_id) in stream_groups:
            task_type_assignments[task_types[sid]] = group_id

        self.herd_assignment = HerdAssignment(worker_groups, task_type_assignments)
        self.allocation_logs.append((current_time, str(self.herd_assignment)))

        if current_time == 0 and ENABLE_MODEL_PREFETCH:
            for worker in self.workers:
                # randomly choose a model to prefetch
                group_model_ids = []
                if self.simulation_name == "shepherd":
                    group_model_ids = self.herd_assignment.group_models[worker.group_id]
                else:
                    group_model_ids = [get_model_id_for_task_type(tt) for tt in task_types]

                if group_model_ids:
                    preloaded_model_id = np.random.choice(list(group_model_ids))
                    preloaded_model = [m for ms in models_by_wf for m in ms if m.model_id == preloaded_model_id][0]
                    print(f"W{worker.worker_id} PRELOADED {preloaded_model_id}")
                    worker.GPU_state.prefetch_model(preloaded_model)
                

    def initialize_model_placement_at_workers(self) -> list[tuple[int, list[int]]]:
        """
            Returns a list of worker configurations (partition size, [model ids])
            based on [ALLOCATION_STRATEGY].
        """

        assert(ALLOCATION_STRATEGY in ["VORTEX", "CUSTOM"])

        all_models = list(self.metadata_service.job_type_models.values())

        # TODO: Make fully configurable

        if ALLOCATION_STRATEGY == "VORTEX":
            models = ["0,2", "1", "3"] # index in all_models[0]
            configs = [6, 12, 24]
            nodes = [i for i in range(TOTAL_NUM_OF_NODES)]

            # ppl1
            throughput = {
                "0,2": {6: 200, 12: 240, 24: 270},
                "1": {24: 45},
                "3": {6: 55, 12: 55, 24: 70},
            }

            # ppl2
            throughput = {
                '0': {12:71, 24: 125},
                '1': {6: 5333, 12: 6083 ,24: 7555},
                '2': {6: 26, 12: 45, 24: 92},
                '3': {12: 3.9, 24: 4.82}
            }

            valid_layouts = [[24], [12, 12], [12, 6, 6], [6, 6, 6, 6]]

            def solve_leximin(locked_lower_bounds):
                m = gp.Model("Leximin_Level")
                x, y = {}, {}
                T_m = {}
                Z = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z")

                for model in models:
                    for node in nodes:
                        for c in configs:
                            if c in throughput[model]:
                                x[model, node, c] = m.addVar(vtype=GRB.INTEGER, name=f"x_{model}_{node}_{c}")

                for node in nodes:
                    for lid, layout in enumerate(valid_layouts):
                        y[node, lid] = m.addVar(vtype=GRB.BINARY, name=f"y_{node}_{lid}")

                for model in models:
                    T_m[model] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"T_{model}")
                    m.addConstr(
                        T_m[model] == gp.quicksum(
                            x[model, node, c] * throughput[model][c]
                            for node in nodes for c in configs if (model, node, c) in x
                        )
                    )

                    if model in locked_lower_bounds:
                        m.addConstr(T_m[model] >= locked_lower_bounds[model])
                    else:
                        m.addConstr(Z <= T_m[model])

                    m.addConstr(gp.quicksum(
                        x[model, node, c] for node in nodes for c in configs if (model, node, c) in x
                    ) >= 1)

                for node in nodes:
                    m.addConstr(gp.quicksum(y[node, lid] for lid in range(len(valid_layouts))) == 1)
                    for c in configs:
                        m.addConstr(
                            gp.quicksum(
                                x[model, node, c] for model in models if (model, node, c) in x
                            ) <= gp.quicksum(
                                y[node, lid] * layout.count(c)
                                for lid, layout in enumerate(valid_layouts)
                            )
                        )

                m.setObjective(Z, GRB.MAXIMIZE)
                m.setParam("OutputFlag", 0)
                m.optimize()

                throughput_vals = {model: T_m[model].X for model in models}
                assignment = {
                    (model, node, c): int(x[model, node, c].X)
                    for model in models for node in nodes for c in configs
                    if (model, node, c) in x and x[model, node, c].X > 0.5
                }
                return throughput_vals, assignment

            # Leximin loop
            locked = {}
            for _ in range(len(models)):
                T_vals, assignment = solve_leximin(locked)
                unlocked = [m for m in models if m not in locked]
                if not unlocked:
                    break
                min_model = min(unlocked, key=lambda m: T_vals[m])
                locked[min_model] = T_vals[min_model]

            # static Gurobi alloc:
            worker_configs = []
            for (model_idxs, node, c), count in assignment.items():
                models = list(map(lambda idx: all_models[0][int(idx)], model_idxs.split(",")))
                for _ in range(count):
                    worker_configs.append((c, models))
                print(f" - Model {model_idxs} assigned {count}x to {node} with MIG {c}GB")
            return worker_configs
        elif ALLOCATION_STRATEGY == "CUSTOM":
            assert(sum(psize for psize, _ in CUSTOM_ALLOCATION) / 24 == TOTAL_NUM_OF_NODES)
            assert(all((psize*(10**6)) in VALID_WORKER_SIZES for psize, _ in CUSTOM_ALLOCATION))
            worker_configs = [(psize, [self.get_model_from_id(mid) for mid in mids])
                              for psize, mids in CUSTOM_ALLOCATION]
            return worker_configs
    
    def initialize_workers(self):
        """
            Creates all workers and initializes model placement.
        """
        if self.job_split == "PER_TASK":
            if ALLOCATION_STRATEGY == "HERD":
                self.run_herd_scheduler(0)
                self.initialize_external_clients()
                if HERD_PERIODICITY > 0:
                    self.event_queue.put(EventOrders(
                        HERD_PERIODICITY, StartHerdSchedulerRerun(self)))
            else:
                worker_configs = self.initialize_model_placement_at_workers()
                for i, config in enumerate(worker_configs):
                    if self.simulation_name == "shepherd":
                        self.workers.append(ShepherdWorker(self, i, config[0], 0))
                    else:
                        self.workers.append(HeftTaskWorker(self, i, config[0]))
                    for model in config[1]:
                        self.metadata_service.add_model_cached_location(model, i, 0)
                        self.workers[-1].GPU_state.prefetch_model(model)
                
                self.herd_assignment = HerdAssignment(
                    [self.workers],
                    { tt: 0 for tt in get_task_types(self.job_types_list) })
                self.initialize_external_clients()

    def initialize_external_clients(self):
        for i, client_config in enumerate(CLIENT_CONFIGS):
            self.external_clients.append(
                ExternalClient(self, i, client_config))
            
    def generate_all_jobs(self):
        """
            Generates exactly the number of jobs specified in [CLIENT_CONFIGS]
            for each client at the designated send rates and places all creation
            events on the event queue.
        """

        job_id = 0
        for client in self.external_clients:
            for job_type in client.job_types:
                curr_time = 0

                curr_send_rate = client.per_job_config[job_type]["SEND_RATES"][0]
                curr_send_rate_idx = 0

                for i in range(client.per_job_config[job_type]["NUM_JOBS"]):
                    if curr_send_rate_idx < len(client.per_job_config[job_type]["SEND_RATES"]) - 1:
                        if i == sum(client.per_job_config[job_type]["SEND_RATE_CHANGE_INTERVALS"][:curr_send_rate_idx+1]):
                            curr_send_rate_idx += 1
                            curr_send_rate = client.per_job_config[job_type]["SEND_RATES"][curr_send_rate_idx]
                    
                    next_job = client.create_job(job_type, job_id, curr_send_rate, curr_time)
                    self.jobs[next_job.id] = next_job
                    curr_time = next_job.create_time + CPU_to_CPU_delay(next_job.tasks[0].input_size)

                    if self.centralized_scheduler:
                        self.event_queue.put(EventOrders(
                            curr_time, JobArrivalAtScheduler(self, next_job)))
                    else:
                        initial_worker_id = client.select_initial_worker_id()
                        self.event_queue.put(EventOrders(
                            curr_time, JobArrivalAtWorker(self, next_job, initial_worker_id)))
                        
                    job_id += 1

    
    """
     --------------    Printing out the simulation results     --------------
    """


    def run_finish(self, last_time, by_job_type=False):
        def _get_jobs_per_sec(jobs):
            if len(jobs) == 0: return 0
            completion_time = max(j.end_time for j in jobs) - min(j.create_time for j in jobs)
            return len(jobs) / completion_time * 1000

        stats_dict = { "clients": [] } 
        for client in self.external_clients:
            stats_dict["clients"].append({})
            for job_type in client.job_types:
                stats_dict["clients"][-1][job_type] = {}
                
                completed_jobs = [j for j in self.jobs.values() if len(j.completed_tasks) == len(j.tasks)]
                stats_dict["clients"][-1][job_type]["throughput_qps"] = _get_jobs_per_sec(completed_jobs)

                if SLO_GRANULARITY == "JOB":
                    nontardy_jobs = [j for j in completed_jobs if j.end_time <= j.create_time + j.slo]
                    stats_dict["clients"][-1][job_type]["goodput_qps"] = _get_jobs_per_sec(nontardy_jobs)

                    tardy_jobs = [j for j in completed_jobs if j.end_time > (j.create_time + j.slo) and \
                                 j.end_time <= j.create_time + (j.slo * (1 + SLO_SLACK))]
                    
                    stats_dict["clients"][-1][job_type][f"jobs_within_{1+SLO_SLACK}slo_per_sec"] = _get_jobs_per_sec(nontardy_jobs + tardy_jobs)
                    stats_dict["clients"][-1][job_type]["total_num_tardy"] = len(tardy_jobs)

                    job_tardiness = [j.end_time - (j.create_time + j.slo) for j in tardy_jobs]
                    stats_dict["clients"][-1][job_type]["median_tardiness_ms"] = np.median(job_tardiness)
                    stats_dict["clients"][-1][job_type]["mean_tardiness_ms"] = np.mean(job_tardiness)
                    stats_dict["clients"][-1][job_type]["std_tardiness_ms"] = np.std(job_tardiness)

                job_latencies = [j.end_time - j.create_time for j in completed_jobs]

                stats_dict["clients"][-1][job_type]["median_latency_ms"] = np.median(job_latencies)
                stats_dict["clients"][-1][job_type]["mean_latency_ms"] = np.mean(job_latencies)
                stats_dict["clients"][-1][job_type]["std_latency_ms"] = np.std(job_latencies)

                stats_dict["clients"][-1][job_type]["total_num_dropped"] = len(set(self.task_drop_log["job_id"]))
                stats_dict["clients"][-1][job_type]["drop_rate_qps"] = len(set(self.task_drop_log["job_id"])) / \
                    (max(j.end_time for j in completed_jobs) - min(j.create_time for j in completed_jobs)) * 1000 \
                    if len(completed_jobs) > 0 else 0
        
        self.sim_stats_log = stats_dict

        # completed_jobs = [j for j in self.jobs.values() if len(j.completed_tasks) == len(j.tasks)]

        # # 1. Get the completed job list to compute statistics 
        # completed_jobs = [j for j in self.jobs.values() if len(
        #     j.completed_tasks) == len(j.tasks)]
        # print_end_jobs(last_time, completed_jobs, self.jobs)
        # # 2. Compute the metrics of interest
        # response_times = [job.end_time -
        #                        job.create_time for job in completed_jobs]
        # slow_down_rate = [(job.end_time - job.create_time) /
        #                        WORKFLOW_LIST[job.job_type_id]["BEST_EXEC_TIME"] for job in completed_jobs]
        # print_response_time(response_times)
        # print_slowdown(slow_down_rate)
        # ADFG_created = []
        # for job in completed_jobs:
        #     if job.ADFG not in ADFG_created:
        #         ADFG_created.append(job.ADFG)
        #         # print(job.job_type_id , job.ADFG)
        # print(".... number of DAG created: {}".format(len(ADFG_created)))
        # print_involved_workers(self.workers)
        # if by_job_type:
        #     response_time_per_type = {}
        #     slow_down_per_type = {}
        #     for job in completed_jobs:
        #         if job.job_type_id not in response_time_per_type:
        #             response_time_per_type[job.job_type_id] = []
        #             slow_down_per_type[job.job_type_id] = []
        #         response_time_per_type[job.job_type_id].append(job.end_time - job.create_time)
        #         slow_down_per_type[job.job_type_id].append((job.end_time - job.create_time) / WORKFLOW_LIST[job.job_type_id]["BEST_EXEC_TIME"])
        #     # print statistics for each job type
        #     print_stats_by_job_type(response_time_per_type, slow_down_per_type)
        if self.produce_breakdown:
            self.produce_time_breakdown_results(completed_jobs)

    def produce_time_breakdown_results(self, completed_jobs):

        dataframe = pd.DataFrame(columns=["job_id", "load_info_staleness", "placement_info_staleness",
                                          "workflow_type", "job_create_time", "scheduler_type", "slowdown", "response_time"])
        dataframe_tasks_log = pd.DataFrame(columns=["workflow_type", "task_id", "worker_id", "task_arrival_time", "task_start_exec_time", "time_to_buffer", "dependency_wait_time",
                                                    "time_spent_in_queue", "model_fetching_time", "execution_time"])

        for index, completed_job in enumerate(completed_jobs):

            if index < 0:  # ignore the first 50 jobs
                continue

            slowdown = (completed_job.end_time - completed_job.create_time) / \
                WORKFLOW_LIST[completed_job.job_type_id]["BEST_EXEC_TIME"]
            response_time = completed_job.end_time - completed_job.create_time
            dataframe.loc[index] = [completed_job.id, LOAD_INFORMATION_STALENESS, PLACEMENT_INFORMATION_STALENESS, completed_job.job_type_id,
                                    completed_job.create_time, self.simulation_name, slowdown, response_time]

        task_index = 0
        for job in completed_jobs:
            for task in job.tasks:
                time_to_buffer = task.log.task_arrival_at_worker_buffer_timestamp - \
                    task.log.job_creation_timestamp
                dependency_wait_time = task.log.task_placed_on_worker_queue_timestamp - \
                    task.log.task_arrival_at_worker_buffer_timestamp
                time_spent_in_queue = task.log.task_front_queue_timestamp - \
                    task.log.task_placed_on_worker_queue_timestamp
                model_fetching_time = task.log.get_model_fetch_time()
                execution_time = task.log.task_execution_end_timestamp - \
                    task.log.task_execution_start_timestamp

                assert time_to_buffer >= 0
                assert dependency_wait_time >= 0
                assert time_spent_in_queue >= 0
                assert model_fetching_time >= 0
                assert execution_time >= 0

                dataframe_tasks_log.loc[task_index] = [job.job_type_id, task.task_id, task.executing_worker_id, task.log.task_arrival_at_worker_buffer_timestamp, 
                                                       task.log.task_execution_start_timestamp,time_to_buffer, dependency_wait_time, 
                                                       time_spent_in_queue, model_fetching_time, execution_time]
                task_index += 1

        self.tasks_logging_times = dataframe_tasks_log
        self.result_to_export = dataframe