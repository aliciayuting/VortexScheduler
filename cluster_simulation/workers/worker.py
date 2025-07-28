from core.config import *
from core.network import *
from core.config import *

from workers.gpu_state import *

import pandas as pd


class Worker(object):
    """ Abstract class representing workers. """

    _abandoned_batches = []

    def __init__(self, simulation, worker_id, total_memory, group_id=-1):
        assert(total_memory in [24, 12, 6])

        self.group_id = group_id
        self.worker_id = worker_id
        self.simulation = simulation
        self.total_memory = total_memory
        
        self.GPU_memory_models = []
        self.GPU_state = GPUState(total_memory * (10**6))

        self.model_history_log = pd.DataFrame(columns=["start_time", "end_time",
                                                       "model_id", "placed_or_evicted"])

    def __hash__(self):
        return hash(self.worker_id)

    def __str__(self):
        return "[Worker_id:{}]".format(self.worker_id)

    def __eq__(self, other):
        if isinstance(other, Worker):
            return self.worker_id == other.worker_id
        return False

    def __ne__(self, other):
        return not (self == other)
    
    def __lt__(self, other):
        return self.worker_id < other.worker_id

    def initial_model_placement(self, model):
        """
        Place the model according to the placement policy
        """
        if (self.used_GPUmemory(0, 0) + model.model_size) < GPU_MEMORY_SIZE:
            self.GPU_memory_models.append(model)
            self.simulation.metadata_service.add_model_cached_location(
                model, self.worker_id, 0)
            return 1
        return 1
    
    """ ----------  BATCH TRACKING AND MANAGEMENT  ---------- """
    
    def evict_batch(self, batch_id: int, time: float):
        evicted_batch = None
        for s in self.GPU_state.state_at(time):
            if s.reserved_batch and s.reserved_batch.id == batch_id:
                evicted_batch = s.reserved_batch
                break
        self.GPU_state.release_busy_model(batch_id, time)
        Worker._abandoned_batches.append(batch_id)
        return evicted_batch

    def did_abandon_batch(self, batch_id: int):
        return batch_id in Worker._abandoned_batches
    
    """ ----------  LOCAL MEMORY MANAGEMENT  ---------- """
    
    def fetch_model(self, model, batch, current_time, exec_time=-1):
        if model == None or self.GPU_state.does_have_idle_copy(model, current_time):
            return 0
        
        fetch_time = 0
        fetch_time = SameMachineCPUtoGPU_delay(model.model_size)

        reserve_until = -1 if exec_time < 0 else (current_time + fetch_time + exec_time)

        self.simulation.metadata_service.add_model_cached_location(
            model, self.worker_id, current_time + fetch_time)
        self.GPU_state.fetch_model(model, current_time, fetch_time, 
                                   reserved_batch=batch, reserve_until=reserve_until)
        
        self.model_history_log.loc[len(self.model_history_log)] = {
            "start_time": current_time,
            "end_time": current_time + fetch_time,
            "model_id": model.model_id,
            "placed_or_evicted": "placed"
        }

        return fetch_time

    LOOKAHEAD_EVICTION = 0
    FCFS_EVICTION = 1

    def evict_models_from_GPU_until(self, current_time: float, min_required_memory: int, policy: int) -> float:
        """
            Evicts models from GPU according to FCFS or lookahead eviction policy until at least
            min_required_memory space is available. Returns time taken to execute model
            evictions. 0 if min_required_memory could not be created.
            Assumes batches run in earliest task arrival order.
        """
        curr_memory = self.GPU_state.available_memory(current_time)
       
        placed_model_states = self.GPU_state.placed_model_states(current_time)
        if policy == self.LOOKAHEAD_EVICTION:
            # TODO: try different eviction policies
            raise NotImplementedError("No lookahead yet")
            # next_models = self.get_next_models(3, current_time)
            # placed_model_states = sorted(
            #     placed_model_states,
            #     key=lambda m: next_models.index(m.model) if m.model in next_models else len(next_models),
            #     reverse=True
            # )

        models_to_evict = []
        for state in placed_model_states:
            if not state.reserved_batch:
                curr_memory += state.model.model_size
                models_to_evict.append(state.model)
                if curr_memory >= min_required_memory:
                    # model_evict_times = list(map(lambda m: SameMachineGPUtoCPU_delay(m.model_size), models_to_evict))
                    # eviction_duration = max(model_evict_times)
                    # full_eviction_end = current_time + eviction_duration

                    # must reserve space to prevent other models from loading in space created here
                    # extra_to_reserve = min_required_memory - sum(m.model_size for m in models_to_evict)
                    # if extra_to_reserve > 0:
                    #     self.GPU_state.reserve_model_space(None, extra_to_reserve, current_time, full_eviction_end)

                    for i in range(len(models_to_evict)):
                        self.simulation.metadata_service.rm_model_cached_location(
                            models_to_evict[i], self.worker_id, current_time)
                        self.GPU_state.evict_model(models_to_evict[i], current_time, 0)
                        
                        self.model_history_log.loc[len(self.model_history_log)] = {
                            "start_time": current_time,
                            "end_time": current_time ,
                            "model_id": models_to_evict[i].model_id,
                            "placed_or_evicted": "evicted"
                        }
                    return 0
        return 0
    
    _CAN_RUN_NOW = 0
    _CAN_RUN_ON_EVICT = 1
    _CANNOT_RUN = 2

    def can_run_task(self, current_time: float, model: Model) -> int:
        """
            Returns _CAN_RUN_NOW if model None, or model is on GPU and not currently in use.
            Returns _CAN_RUN_ON_EVICT if model can be loaded onto the GPU upon evicting
            unused models.
            Returns _CANNOT_RUN otherwise.
        """
        if model == None or self.GPU_state.does_have_idle_copy(model, current_time):
            return self._CAN_RUN_NOW
        
        # cannot load additional copies of the same model
        if any(map(lambda s: s.model == model, self.GPU_state.state_at(current_time))):
            return self._CANNOT_RUN
        
        if self.GPU_state.can_fetch_model(model, current_time):
            return self._CAN_RUN_NOW
        
        if self.GPU_state.can_fetch_model_on_eviction(model, current_time):
            return self._CAN_RUN_ON_EVICT
        
        return self._CANNOT_RUN

    # ------------------------- cached model history update helper functions ---------------
    def get_history(self, history, current_time, info_staleness) -> list:
        delayed_time = current_time - info_staleness
        last_index = len(history) - 1
        while last_index >= 0:
            if history[last_index][0] <= delayed_time:
                return history[last_index][1].copy()
            last_index -= 1  # check the previous one
        return []