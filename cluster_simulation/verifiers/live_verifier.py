from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from workers.worker import Worker
from client.client import Client

from core.data_models.workflow import Workflow
from core.batch import Batch
from core.network import *

from uuid import UUID

import core.configs.gen_config as gcfg
import core.configs.workflow_config as wcfg
import core.configs.model_config as mcfg

import numpy as np

from scipy.stats import norm



class LiveVerifier(EventListener):

    def __init__(self, em: EventManager, clients: dict[UUID, Client], workers: dict[UUID, Worker],
                 scheduler_worker_id: UUID, is_centralized: bool):
        
        super().__init__(Agent.VERIFIER)

        self.verify_worker_configs(workers)

        self.is_centralized = is_centralized
        self.scheduler_worker_id = scheduler_worker_id
        self.clients = clients
        self.workers = workers
        self.em = em

        # init state to use for verification
        self.instance_states: dict[UUID, list[tuple[int, int]]] = {}
        self.model_id_to_instances: dict[int, list[UUID]] = {}
        self.worker_id_to_instances: dict[UUID, list[UUID]] = {}
        for w in workers.values():
            self.worker_id_to_instances[w.id] = []
            for s in w.GPU_state.state_at(0):
                if s.model.data.id not in self.model_id_to_instances:
                    self.model_id_to_instances[s.model.data.id] = []
                self.model_id_to_instances[s.model.data.id].append(s.model.id)
                self.instance_states[s.model.id] = None
                self.worker_id_to_instances[w.id].append(s.model.id)

        # track significant deviations from expected batch exec times
        self.sampled_anomalies = 0
        self.total_samples = 0

        self.task_send_queue: dict[tuple[int, int], tuple[float, UUID]] = {} # -> (expected arrival time, worker ID)
        self.task_exec_queue: dict[tuple[int, int], tuple[float, UUID]] = {} # -> (expected exec start time, instance ID)

        self.em.register_listener(self, {
            EVENT_TYPES[EventIds.JOB_SENT_TO_SCHEDULER],
            EVENT_TYPES[EventIds.JOB_ARRIVAL_AT_SCHEDULER],
            EVENT_TYPES[EventIds.TASKS_ARRIVAL_AT_SCHEDULER],
            
            EVENT_TYPES[EventIds.TASKS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ASSIGNED_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_SENT_TO_WORKER],
            EVENT_TYPES[EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER],

            EVENT_TYPES[EventIds.JOBS_DROPPED],

            EVENT_TYPES[EventIds.BATCH_STARTED_AT_WORKER],
            EVENT_TYPES[EventIds.BATCH_FINISHED_AT_WORKER],
            EVENT_TYPES[EventIds.RESPONSE_SENT_TO_CLIENT],
            EVENT_TYPES[EventIds.RESPONSE_RECEIVED_AT_CLIENT]
        })

        # (job ID, task ID) -> {arrival at sched, arrival at worker, exec start, exec end}
        self.task_log: dict[tuple[int, int], dict[str, int]] = {}
        

    def on_event(self, event: Event):
        workflow_ids = set([k for ccfg in gcfg.CLIENT_CONFIGS for k in ccfg.keys()])
        workflow_cfgs = {wwcfg["JOB_TYPE"]: wwcfg for wwcfg in wcfg.WORKFLOW_LIST 
                         if wwcfg["JOB_TYPE"] in workflow_ids}
        
        self.verify_network_delays(event)

        # tasks become available for execution EXCLUSIVELY
        # in input and output arrival events
        if event.type.id == EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER:
            if gcfg.DISPATCH_POLICY == "SHEPHERD":
                # Shepherd batches everything at scheduler, no worker queues
                assert("force_instance_id" in event.kwargs)
                assert(self.instance_states[event.kwargs["force_instance_id"]] == None)

                # assigned instance should be a copy of the required model for all tasks
                assert(all(event.kwargs["force_instance_id"] in self.model_id_to_instances[t.model_data.id] 
                           for t in event.kwargs["tasks"]))
                
                # expect tasks to be executed immediately
                for task in event.kwargs["tasks"]:
                    self.task_exec_queue[(task.job.id, task.task_id)] = (
                        event.time, event.kwargs["force_instance_id"])
            
            elif gcfg.DISPATCH_POLICY == "ROUND_ROBIN":
                for task in event.kwargs["tasks"]:
                    relevant_instances = self.model_id_to_instances[task.model_data.id]
                    
                    # assigned worker must have at least 1 copy of required model
                    assert(any(iid in self.worker_id_to_instances[event.kwargs["to_worker_id"]]
                               for iid in relevant_instances))
                    
                    # if an idle instance exists, task should be executed immediately
                    for iid in self.worker_id_to_instances[event.kwargs["to_worker_id"]]:
                        if iid in relevant_instances and self.instance_states[iid] == None:
                            self.task_exec_queue[(task.job.id, task.task_id)] = (event.time, None)
                    
        elif event.type.id == EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER:
            pass

        elif event.type.id == EventIds.BATCH_STARTED_AT_WORKER:
            batch: Batch = event.kwargs["batch"]

            # batch size never exceeds max batch size
            assert(batch.size() <= mcfg.MODELS[batch.model_data.id]["MAX_BATCH_SIZE"])
            
            # all tasks use the same model
            assert(all(workflow_cfgs[t.job.job_type_id]["TASKS"][t.task_id]["MODEL_ID"] == batch.model_data.id
                       for t in batch.tasks))
            
            # instance is currently idle
            assert(self.instance_states[event.kwargs["model_instance_id"]] == None)

            # instance is correct
            assert(event.kwargs["model_instance_id"] in self.model_id_to_instances[batch.model_data.id])
            assert(event.kwargs["model_instance_id"] in self.worker_id_to_instances[event.kwargs["worker_id"]])
            
            for t in batch.tasks:
                # for all tasks, dependencies should all have been completed
                for prev_id in workflow_cfgs[t.job.job_type_id]["TASKS"][t.task_id]["PREV_TASK_INDEX"]:
                    assert((t.job.id, prev_id) in self.task_log)
                    assert(self.task_log[(t.job.id, prev_id)]["exec_end_time"] <= event.time and \
                           self.task_log[(t.job.id, prev_id)]["exec_end_time"] >= 0)
                
                # for all tasks, under LARGEST_FEASIBLE policy should NOT
                # violate SLO
                if gcfg.BATCH_POLICY == "LARGEST_FEASIBLE":
                    deadline = self.clients[t.job.client_id][3]
                    worker = self.workers[event.kwargs["worker_id"]]
                    expected_end_time = event.time + t.model_data.batch_exec_times[worker.total_memory_gb][batch.size()]
                    assert(expected_end_time <= deadline)

                # for all tasks, should be anticipated
                if (t.job.id, t.task_id) in self.task_exec_queue:
                    assert((t.job.id, t.task_id) in self.task_exec_queue)
                    
                    if gcfg.DISPATCH_POLICY == "SHEPHERD":
                        assert(self.task_exec_queue[(t.job.id, t.task_id)] == 
                               (event.time, event.kwargs["model_instance_id"]))
                    # else:
                    #     print(self.task_exec_queue[(t.job.id, t.task_id)])
                    #     assert(self.task_exec_queue[(t.job.id, t.task_id)] == 
                    #            (event.time, None))
                    
                    self.task_exec_queue.pop((t.job.id, t.task_id))

                self.task_log[(t.job.id, t.task_id)] = {
                    "exec_start_time": event.time,
                    "exec_end_time": -1
                }
            
            self.instance_states[event.kwargs["model_instance_id"]] = [(t.job.id, t.task_id) for t in batch.tasks]

        elif event.type.id == EventIds.BATCH_FINISHED_AT_WORKER:
            batch: Batch = event.kwargs["batch"]

            assert(self.instance_states[event.kwargs["model_instance_id"]] == 
                   [(t.job.id, t.task_id) for t in batch.tasks])

            for t in batch.tasks:
                assert((t.job.id, t.task_id) in self.task_log)
                assert(self.task_log[(t.job.id, t.task_id)]["exec_start_time"] < event.time)

                worker_size = self.workers[event.kwargs["worker_id"]].total_memory_gb
                expected_batch_exec_time = mcfg.MODELS[batch.model_data.id]["MIG_BATCH_EXEC_TIMES"][worker_size][batch.size()]
                actual_batch_exec_time = event.time - self.task_log[(t.job.id, t.task_id)]["exec_start_time"]
                expected_transfer_time = sum(workflow_cfgs[t1.job.job_type_id]["TASKS"][t1.task_id]["INPUT_SIZE"] 
                                             for t1 in batch.tasks) / 64000

                if not self.is_sampled_correctly(actual_batch_exec_time,
                                                 expected_batch_exec_time + expected_transfer_time,
                                                 mcfg.MODELS[batch.model_data.id]["EXEC_TIME_CVS"][worker_size]):
                    self.sampled_anomalies += 1
                self.total_samples += 1

                self.task_log[(t.job.id, t.task_id)]["exec_end_time"] = event.time
            
            self.instance_states[event.kwargs["model_instance_id"]] = None

        elif event.type.id == EventIds.RESPONSE_SENT_TO_CLIENT:
            job: Job = event.kwargs["job"]

            assert(all(self.task_log[(job.id, task_id)]["exec_end_time"] >= 0 and \
                       self.task_log[(job.id, task_id)]["exec_end_time"] <= event.time 
                       for task_id in range(len(workflow_cfgs[job.job_type_id]["TASKS"]))))
            

    def is_sampled_correctly(self, x, mu, cv):
        """Calculate two sided p-value to test probability of being sampled
        from a given Normal distribution.
        """
        sigma = cv * mu
        z = (x - mu) / sigma
        p = 2 * norm.sf(abs(z))  # two-sided
        return p >= 0.05
    

    def verify_worker_configs(self, workers: dict[UUID, Worker]):
        if gcfg.ALLOCATION_STRATEGY == "CUSTOM":
            worker_id_to_cfg: dict[UUID, tuple[int, list[int]]] = {}
            for worker_id, worker in workers.items():
                worker_id_to_cfg[worker_id] = (worker.total_memory_gb,
                                               [s.model.data.id for s in worker.GPU_state.state_at(0)])

            required_alloc = gcfg.CUSTOM_ALLOCATION.copy()
            for worker_id, cfg in worker_id_to_cfg.items():
                assert(cfg in required_alloc)
                required_alloc.remove(cfg)
            
            assert(len(required_alloc) == 0)
    

    def verify_network_delays(self, event: Event):
        if event.type.id in [EventIds.JOB_SENT_TO_SCHEDULER,
                             EventIds.TASKS_INPUTS_SENT_TO_WORKER,
                             EventIds.TASKS_OUTPUTS_SENT_TO_WORKER,
                             EventIds.RESPONSE_SENT_TO_CLIENT]:

            tasks: list[Task] = []
            expected_worker_id: UUID = None
            if event.type.id == EventIds.JOB_SENT_TO_SCHEDULER:
                tasks = [t for t in event.kwargs["job"].tasks if len(t.required_task_ids) == 0]
                expected_worker_id = self.scheduler_worker_id
            elif event.type.id == EventIds.RESPONSE_SENT_TO_CLIENT:
                tasks = [t for t in event.kwargs["job"].tasks if len(t.next_task_ids) == 0]
            else:
                tasks = event.kwargs["tasks"]
                expected_worker_id = event.kwargs["to_worker_id"]

            expected_arrival_time: float = 0
            if gcfg.ENABLE_NETWORKING_DELAYS:
                if event.type.id == EventIds.JOB_SENT_TO_SCHEDULER:
                    expected_arrival_time = CPU_to_CPU_delay(sum([t.input_size for t in tasks]))
                elif event.type.id == EventIds.RESPONSE_SENT_TO_CLIENT:
                    expected_arrival_time = CPU_to_CPU_delay(sum([t.result_size for t in tasks]))
                elif event.kwargs["to_worker_id"] != event.kwargs["from_worker_id"]:
                    if event.type.id == EventIds.TASKS_INPUTS_SENT_TO_WORKER:
                        if self.is_centralized:
                            # sched -> worker no cost if decentral
                            expected_arrival_time = CPU_to_CPU_delay(sum([t.input_size for t in tasks]))
                    else:
                        expected_arrival_time = CPU_to_CPU_delay(sum([t.result_size for t in tasks]))
            
            for task in tasks:
                if (task.job.id, task.task_id) not in self.task_send_queue:
                    self.task_send_queue[(task.job.id, task.task_id)] = []
                
                self.task_send_queue[(task.job.id, task.task_id)].append(
                    (event.time + expected_arrival_time, expected_worker_id))
                
        elif event.type.id in [EventIds.JOB_ARRIVAL_AT_SCHEDULER, 
                               EventIds.TASKS_INPUTS_ARRIVAL_AT_WORKER, 
                               EventIds.TASKS_OUTPUTS_ARRIVAL_AT_WORKER,
                               EventIds.RESPONSE_RECEIVED_AT_CLIENT]:
            
            tasks: list[Task] = []
            receiving_worker_id: UUID = None
            if event.type.id == EventIds.JOB_ARRIVAL_AT_SCHEDULER:
                tasks = [t for t in event.kwargs["job"].tasks if len(t.required_task_ids) == 0]
                receiving_worker_id = self.scheduler_worker_id
            elif event.type.id == EventIds.RESPONSE_RECEIVED_AT_CLIENT:
                tasks = [t for t in event.kwargs["job"].tasks if len(t.next_task_ids) == 0]
            else: 
                tasks = event.kwargs["tasks"]
                receiving_worker_id = event.kwargs["to_worker_id"]

            for task in tasks:
                assert((task.job.id, task.task_id) in self.task_send_queue and \
                       len(self.task_send_queue[(task.job.id, task.task_id)]) > 0)
                assert((event.time, receiving_worker_id) in self.task_send_queue[(task.job.id, task.task_id)])
                self.task_send_queue[(task.job.id, task.task_id)].remove((event.time, receiving_worker_id))


    def verify_on_sim_end(self):
        assert(sum(len(v) for v in self.task_send_queue.values()) == 0)
        assert(len(self.task_exec_queue.keys()) == 0)