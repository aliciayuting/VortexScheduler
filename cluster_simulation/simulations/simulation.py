import os
import pandas as pd

import core.configs.gen_config as gcfg
import core.configs.workflow_config as wcfg
import core.configs.model_config as mcfg

from core.data_models.workflow import Workflow
from core.data_models.model_data import ModelData
from core.workload import (get_client_workloads, get_workflow_peak_rates,
                           get_workflow_slos)

from client.client import Client
from network.network import Network
from schedulers.shepherd_scheduler import ShepherdScheduler
from schedulers.central_round_robin_scheduler import CentralRoundRobinScheduler
from schedulers.decentral_round_robin_scheduler import DecentralRoundRobinScheduler
from workers.worker import Worker

from core.allocation import ModelAllocation
from schedulers.algo.vortex_planner_algo import VortexPlanner
from schedulers.algo.nexus_algo import NexusSLOSplitter

from verifiers.live_verifier import LiveVerifier
from verifiers.log_verifier import LogVerifier

from sim_logging.logger import Logger

from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from uuid import uuid4


class Simulation:

    def __init__(self, centralized: bool, out_path: str):
        self.out_path = out_path
        self.em = EventManager()
        self.is_centralized = centralized

        self.models = self._generate_models()
        self.workflows = self._generate_workflows()
        self.clients = self._generate_clients()

        self.allocation = self._generate_model_allocation()
        self.workers = self._generate_workers()

        self._assign_nexus_task_slos()

        self.scheduler = None
        scheduler_worker_id = None
        if centralized:
            scheduler_worker_id = list(self.workers.keys())[0]

            if gcfg.DISPATCH_POLICY == "SHEPHERD":
                self.scheduler = ShepherdScheduler(
                    self.em, self.workers, self.workflows, scheduler_worker_id)
            
            elif gcfg.DISPATCH_POLICY == "ROUND_ROBIN":
                self.scheduler = CentralRoundRobinScheduler(
                    self.em, self.workers, self.workflows, scheduler_worker_id)

            else:
                assert("Unknown central dispatch policy")

        else:
            if gcfg.DISPATCH_POLICY == "ROUND_ROBIN":
                self.scheduler = DecentralRoundRobinScheduler(
                    self.em, self.workers, self.workflows)
                
            else:
                assert("Unknown decentral dispatch policy")


        self.network = Network(self.em, scheduler_worker_id)
        self.verifier = LiveVerifier(self.em, 
                                     {c.id: c for c in self.clients}, 
                                     self.workers,
                                     scheduler_worker_id,
                                     centralized)
        self.logger = Logger(self.em, self.workers)


    def _generate_clients(self) -> list[Client]:
        """Generates clients from client config and enqueues all
        job creation/send events.

        Returns:
            clients: List of generated clients
        """
        clients: list[Client] = []
        created_job_count = 0
        for wid, workload, slo in get_client_workloads():
            client = Client(uuid4(), self.em)
            clients.append(client)

            client.generate_jobs(self.workflows[wid], workload, 0, slo, created_job_count)

            # job IDs are global, so the next client picks up where this one left off
            created_job_count += len(client.jobs)

        return clients
    

    def _generate_models(self) -> dict[int, ModelData]:
        """Initializes ModelData objects for all models used by jobs issued by configured
        clients.

        Returns:
            models: Map of model ID -> abstract model representation
        """
        all_workflow_ids = set([k for c in gcfg.CLIENT_CONFIGS for k in c.keys()])
        model_ids: set[int] = set()
        for cfg in wcfg.WORKFLOW_LIST:
            if cfg["JOB_TYPE"] not in all_workflow_ids:
                continue
            
            for task_cfg in cfg["TASKS"]:
                if task_cfg["MODEL_ID"] >= 0:
                    model_ids.add(task_cfg["MODEL_ID"])

        return {id: ModelData(id,
                              mcfg.MODELS[id]["MODEL_SIZE"],
                              mcfg.MODELS[id]["MAX_BATCH_SIZE"],
                              mcfg.MODELS[id]["MIG_BATCH_EXEC_TIMES"],
                              mcfg.MODELS[id]["EXEC_TIME_CVS"]) 
                              for id in model_ids}

    
    def _generate_workflows(self) -> dict[int, Workflow]:
        """Initializes Workflow objects for all workflow types required by
        configured clients.

        Returns:
            workflows: Map of workflow ID -> abstract workflow representation
        """
        return {
            cfg["JOB_TYPE"] : Workflow(cfg, self.models, gcfg.SLO_TYPE) for cfg in wcfg.WORKFLOW_LIST
            if cfg["JOB_TYPE"] in set([k for c in gcfg.CLIENT_CONFIGS for k in c.keys()])}


    def _generate_model_allocation(self) -> ModelAllocation:
        if gcfg.ALLOCATION_STRATEGY == "CUSTOM":
            assert(all((psize*(10**6)) in gcfg.VALID_WORKER_SIZES for psize, _ in gcfg.CUSTOM_ALLOCATION))
            return ModelAllocation(
                self,
                {uuid4(): cfg for cfg in gcfg.CUSTOM_ALLOCATION},
                reset_batch_sizes=False)
        
        elif gcfg.ALLOCATION_STRATEGY == "INFERLINE":
            # TODO: add support for multitenant
            assert(len(self.workflows) == 1)
            assert(len(gcfg.CLIENT_CONFIGS) == 1 and len(gcfg.CLIENT_CONFIGS[0].keys()) == 1)

            slo = list(gcfg.CLIENT_CONFIGS[0].values())[0]["SLO"]
            alloc = self.inferline.planner_minimize_cost(self, 0, self.workflows[0], slo)

            return alloc
        
        elif gcfg.ALLOCATION_STRATEGY == "VORTEX":
            # NOTE: no support for multitenant yet
            assert(len(self.workflows) == 1)
            assert(len(gcfg.CLIENT_CONFIGS) == 1 and len(gcfg.CLIENT_CONFIGS[0].keys()) == 1)

            slo = list(gcfg.CLIENT_CONFIGS[0].values())[0]["SLO"]
            return VortexPlanner.get_allocation(self,
                                                (gcfg.MIN_NUM_NODES + gcfg.MAX_NUM_NODES) // 2,
                                                self.workflows[0],
                                                slo)


    def _assign_nexus_task_slos(self):
        """Splits each configured workflow's job SLO across its pipeline stages,
        giving every task its own deadline. Runs once at setup, after model
        placement is known, since the split depends on how many workers host each
        model. No-op unless SLO_TYPE is NEXUS.

        One Workflow object is shared by every client sending it, so the split is
        planned for the heaviest load that workflow can see: the sum of its
        clients' peak send rates.
        """
        if gcfg.SLO_TYPE != "NEXUS":
            return

        peak_rates = get_workflow_peak_rates()
        slos = get_workflow_slos()

        for wid, workflow in self.workflows.items():
            arrival_rates = NexusSLOSplitter.get_task_arrival_rates(
                workflow, peak_rates[wid], self.workers)
            worker_sizes = NexusSLOSplitter.get_task_worker_sizes(workflow, self.workers)

            workflow.assign_task_slos(
                NexusSLOSplitter.generate_task_slos(
                    workflow, slos[wid], arrival_rates, worker_sizes),
                slos[wid])


    def _generate_workers(self) -> dict[UUID, Worker]:
        """Generates and initializes model placements for initial worker objects.

        Returns:
            workers: Map of worker ID -> worker object
        """
        workers = {}
        for (wid, _) in self.allocation.worker_ids_by_create_time:
            cfg = self.allocation.worker_cfgs[wid]
            worker = Worker(wid, self.em, cfg[0], 0, self.is_centralized)
            workers[wid] = worker
            for mid in cfg[1]:
                self.models[mid].max_batch_size = self.allocation.models[mid].max_batch_size
                instance_id = workers[wid].GPU_state.prefetch_model(self.models[mid])
        return workers


    def run(self):
        while self.em.has_events():
            self.em.process_next_event()
        
        if gcfg.ENABLE_LIVE_VERIFICATION and self.verifier.total_samples > 0:
            sampled_anomalies = self.verifier.sampled_anomalies / self.verifier.total_samples
            if sampled_anomalies > 0.05:
                RED_BOLD = "\033[1;31m"
                RESET = "\033[0m"
                print(f"{RED_BOLD}[VERIFIER WARNING] {sampled_anomalies * 100:.2f}% of batch execution time samples ({self.verifier.sampled_anomalies}/{self.verifier.total_samples}) showed significant deviation (p < 0.05) given configured mean and CV{RESET}")

        self.logger.finalize()

        self.logger.task_log.to_csv(os.path.join(self.out_path, "task_log.csv"))
        self.logger.worker_log.to_csv(os.path.join(self.out_path, "worker_batch_log.csv"))

        self._get_client_data()
        self._postprocess_idle_times()
        self._postprocess_nonexec_delays()

        if gcfg.PRODUCE_EVENT_LOG:
            self.em.event_log.to_csv(os.path.join(self.out_path, "event_log.csv"))
        
        self._produce_agent_keys()
        self._produce_nexus_slo_split_log()
        self._produce_admission_control_log()

        if gcfg.ENABLE_LIVE_VERIFICATION:
            self.verifier.verify_on_sim_end()

        if gcfg.ENABLE_VERIFICATION:
            log_verifier = LogVerifier(None, self.logger.task_log, self.logger.worker_log,
                                       self.worker_config_log, self.is_centralized, gcfg, mcfg, wcfg)
            log_verifier.run()


    def _produce_agent_keys(self):
        worker_log = pd.DataFrame(columns=["worker_id", "worker_creation_timestamp", "instance_id", "model_id",
                                           "instance_loaded_timestamp"])
        for worker in self.workers.values():
            for s in worker.GPU_state.state_at(0):
                worker_log.loc[len(worker_log)] = [worker.id, worker.create_time, s.model.id, s.model.data.id,
                                                   s.model.active_from]

        worker_log.to_csv(os.path.join(self.out_path, "worker_config_log.csv"))
        self.worker_config_log = worker_log
        # TODO: client log


    def _produce_admission_control_log(self):
        """Writes what admission control did to each workflow: the capacity it
        estimated, the rate and burst size it allowed, and how many jobs it turned
        away.
        """
        stats = self.scheduler.admission_controller.get_stats()
        pd.DataFrame(stats).to_csv(os.path.join(self.out_path, "admission_control.csv"))


    def _produce_nexus_slo_split_log(self):
        """Writes the per-stage SLO split that NexusSLOSplitter produced for each
        workflow. No-op unless SLO_TYPE is NEXUS, since no split exists otherwise.
        """
        if gcfg.SLO_TYPE != "NEXUS":
            return

        slo_df = pd.DataFrame(columns=["workflow_id", "task_id", "model_id", "job_slo",
                                       "task_slo", "task_deadline_offset",
                                       "max_batch_size", "worker_size", "min_exec_time"])
        for workflow in sorted(self.workflows.values(), key=lambda w: w.id):
            if not workflow.task_deadline_offsets:
                continue

            worker_sizes = NexusSLOSplitter.get_task_worker_sizes(workflow, self.workers)

            for task_id, task in sorted(workflow.tasks.items()):
                slo_df.loc[len(slo_df)] = {
                    "workflow_id": workflow.id,
                    "task_id": task_id,
                    "model_id": task.model_data.id,
                    "job_slo": workflow.job_slo,
                    "task_slo": workflow.task_slos[task_id],
                    "task_deadline_offset": workflow.task_deadline_offsets[task_id],
                    "max_batch_size": workflow.task_max_batch_sizes[task_id],
                    "worker_size": worker_sizes[task_id],
                    "min_exec_time": task.model_data.batch_exec_times[worker_sizes[task_id]][1]
                }

        slo_df.to_csv(os.path.join(self.out_path, "nexus_slo_split.csv"))


    def _postprocess_nonexec_delays(self):
        delays_df = pd.DataFrame(columns=["workflow_id", "task_id", "mean_queueing_time", "std_queueing_time",
                                          "mean_dispatch_time", "std_dispatch_time"])
        for w in sorted(set(self.logger.task_log["workflow_id"])):
            df = self.logger.task_log[self.logger.task_log["workflow_id"]==w]
            for t in sorted(set(df["task_id"])):
                tdf = df[df["task_id"]==t]
                queueing_times = tdf["execution_start_timestamp"] - tdf["arrival_at_worker_timestamp"]
                dispatch_times = tdf["arrival_at_worker_timestamp"] - tdf["last_dep_dispatch_timestamp"]

                delays_df.loc[len(delays_df)] = [w, t, queueing_times.mean(), queueing_times.std(),
                                                 dispatch_times.mean(), dispatch_times.std()]
        
        delays_df.to_csv(os.path.join(self.out_path, "nonexec_delays.csv"))

    def _postprocess_idle_times(self):
        idle_df = pd.DataFrame(columns=["worker_id", "instance_id", "model_id", "idle_time_s", "idle_percent_worker_lifetime", 
                                        "mean_batch_size", "std_batch_size"])
        last_task_exec_end = self.logger.worker_log["execution_end_timestamp"].max()
        for instance_id in set(self.logger.worker_log["instance_id"]):
            df = self.logger.worker_log[self.logger.worker_log["instance_id"]==instance_id]
            idle_time = 0
            last_exec_end = -1
            for i, row in df.iterrows():
                if last_exec_end < 0:
                    idle_time += row["execution_start_timestamp"]
                else:
                    idle_time += row["execution_start_timestamp"] - last_exec_end
                
                last_exec_end = row["execution_end_timestamp"]
            
            model_id = df["model_id"].iloc[0]
            idle_df.loc[len(idle_df)] = [df["worker_id"].iloc[0], instance_id, model_id, 
                                         idle_time, idle_time / last_task_exec_end * 100,
                                         df["batch_size"].mean(), df["batch_size"].std()]

        idle_df.to_csv(os.path.join(self.out_path, "model_instance_idle_times.csv"))

    def _get_client_data(self):
        jobs_df = pd.DataFrame(columns=["client_id", "workflow_id", "job_id", "was_completed",
                                        "deadline", "create_time", "response_time"])
        for client in self.clients:
            for jid, (create_time, finish_time, was_completed, deadline, job) in client.jobs.items():
                jobs_df.loc[len(jobs_df)] = {
                    "client_id": client.id,
                    "workflow_id": job.job_type_id,
                    "job_id": jid,
                    "was_completed": was_completed,
                    "deadline": deadline,
                    "create_time": create_time,
                    "response_time": finish_time - create_time
                }
        jobs_df.to_csv(os.path.join(self.out_path, "job_log.csv"))
