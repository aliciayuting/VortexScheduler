import copy

from core.data_models.model_data import ModelData
from core.model import Model
from core.batch import Batch

import numpy as np


class ModelState:
    """Wrapper class representing a Model placed on a GPU.
    """
    
    PLACED = 0
    PRE_FETCH = 1 # reserved for a model that will be fetched
    IN_FETCH = 2
    IN_EVICT = 3

    def __init__(self, model: Model, state: int, reserved_batch: Batch=None, reserved_until=-1, size=0):
        self.model = model
        self.size = size if size > 0 else model.data.size
        self.state = state
        self.reserved_batch = reserved_batch
        self.reserved_until = reserved_until

    def __eq__(self, value):
        return type(value) == ModelState and self.model.id == value.model.id
    
    def __str__(self):
        executing_batch = f'EXECUTING BATCH {self.reserved_batch.id}' if self.reserved_batch else 'NOT IN USE'
        return f"<[{self._state_to_str()}] [{executing_batch}] Model ID: {self.model.data.id if self.model else -1}, Instance ID: {self.model.id if self.model else 'N/A'}>"
    
    def __repr__(self):
        return self.__str__()
    
    def _state_to_str(self):
        if self.state == self.PLACED: return "Placed"
        elif self.state == self.IN_FETCH: return "Fetching"
        elif self.state == self.IN_EVICT: return "Evicting"
        elif self.state == self.PRE_FETCH: return "Reserved"

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied

        copied.model = self.model
        copied.reserved_batch = self.reserved_batch
        copied.reserved_until = copy.deepcopy(self.reserved_until, memo)
        copied.size = copy.deepcopy(self.size, memo)
        copied.state = copy.deepcopy(self.state, memo)

        return copied


class GPUState(object):
    """
        State of a single GPU at any given time.
    """
    
    def __init__(self, total_memory: int):
        # sorted (asc) list of GPU states [(time, [model states])]
        self._model_states = []
        self._total_memory = total_memory

    def reserved_memory(self, time: float) -> float:
        """
            Returns the total GPU memory that is currently in use, either
            for currently placed models, models that are being fetched,
            models that are being evicted, or models that will be fetched.
        """
        return sum(state.size for state in self.state_at(time))
    
    def available_memory(self, time: float) -> float:
        """
            Returns total GPU memory that is not reserved (see reserved_memory).
        """
        return self._total_memory - self.reserved_memory(time)

    def can_fetch_model(self, model_data: ModelData, time: float) -> bool:
        """
            Returns True if a new copy of [model] can be fetched to the
            GPU as is with no evictions.
        """
        return self.available_memory(time) >= model_data.size
    
    def can_fetch_model_on_eviction(self, model_data: ModelData, time: float) -> bool:
        """
            Return True if a new copy of [model] can be fetched to the GPU
            upon evicting some number of placed models not in use.
        """
        # cannot use space occupied by models currently being fetched/evicted or used
        return (self.available_memory(time) + \
                sum(state.size for state in self.state_at(time)
                    if state.state == ModelState.PLACED and not state.reserved_batch)) >= model_data.size

    def prefetch_model(self, model_data: ModelData) -> str:
        """Preload given model to GPU for time 0 with no fetch cost.

        Args:
            model_data: Model to load

        Returns:
            instance_id: ID of loaded copy of model
        """
        assert(self.can_fetch_model(model_data, 0))

        model = model_data.create_instance(0, 0)
        if len(self._model_states) == 0:
            self._model_states.append((0, [ModelState(model, ModelState.PLACED)]))
        else:
            self._model_states[0][1].append(ModelState(model, ModelState.PLACED))
        return model.id
    
    def _insert_state_marker(self, marker_time: float, at_marker_modify, post_marker_modify):
        """
            Internal helper to update states at exactly [marker_time] with 
            [at_marker_modify: (time, old_states) -> new_states] and states
            after [marker_time] with 
            [post_marker_modify: (time, old_states) -> new_states].
        """
        did_add_marker = False
        for i in range(len(self._model_states)-1, -1, -1):
            (timestamp, states) = self._model_states[i]
            if timestamp == marker_time:
                at_marker_modify(timestamp, states)
                did_add_marker = True
            elif timestamp < marker_time:
                if not did_add_marker:
                    state_copy = copy.deepcopy(states)
                    at_marker_modify(timestamp, state_copy)
                    self._model_states.insert(i+1, (marker_time, state_copy))
                    did_add_marker = True
                return
            else:
                post_marker_modify(timestamp, states)

        if not did_add_marker:
            states = []
            at_marker_modify(marker_time, states)
            self._model_states.insert(0, (marker_time, states))

    def fetch_model(self, model_data: ModelData, start_time: float, fetch_time: float, reserved_batch: Batch=None, reserve_until=-1) -> str:
        """Fetch a new copy of a given model if there is enough available_space().

        Args:
            model_data: Model to load
            start_time: Time to start fetch
            fetch_time: Duration of fetch
            reserved_batch: Batch to reserve model for
            reserve_until: Time to reserve model until
        
        Returns:
            instance_id: ID of fetched instance
        """
        assert(model_data != None)
        assert(self.can_fetch_model(model_data, start_time))

        fetch_end_time = start_time + fetch_time

        model = model_data.create_instance(start_time, fetch_time)

        if len(self._model_states) == 0:
            # mark when fetch begins and ends
            self._model_states.append((start_time, [ModelState(model, ModelState.IN_FETCH,
                                                               reserved_batch=reserved_batch,
                                                               reserved_until=reserve_until)]))
            self._model_states.append((fetch_end_time, [ModelState(model, ModelState.PLACED, 
                                                                   reserved_batch=reserved_batch,
                                                                   reserved_until=reserve_until)]))
            return model.id
        
        # add fetch end marker
        self._insert_state_marker(fetch_end_time,
                                  lambda _, states: states.append(ModelState(model, ModelState.PLACED,
                                                                             reserved_batch=reserved_batch, 
                                                                             reserved_until=reserve_until)),
                                  lambda _, states: states.append(ModelState(model, ModelState.PLACED,
                                                                             reserved_batch=reserved_batch,
                                                                             reserved_until=reserve_until)))
        
        # add fetch start marker
        self._insert_state_marker(start_time,
                                  lambda _, states: states.append(ModelState(model, ModelState.IN_FETCH,
                                                                             reserved_batch=reserved_batch,
                                                                             reserved_until=reserve_until)),
                                  lambda t, states: states.append(ModelState(model, ModelState.IN_FETCH,
                                                                             reserved_batch=reserved_batch,
                                                                             reserved_until=reserve_until)) if t < fetch_end_time else None)
        
        return model.id
    
    def evict_model_instance(self, instance_id: str, start_time: float, evict_time: float=0) -> Batch | None:
        """Evicts a specific copy of a model.

        Args:
            instance_id: ID of model instance to evict
            start_time: When to start eviction
            evict_time: Duration of eviction

        Returns:
            evicted_batch: If instance was executing a batch, return the evicted batch
        """
        evicted_state = [s.model.id == instance_id and s.state == ModelState.PLACED
                         for s in self.state_at(start_time)]
        
        assert(len(evicted_state) == 1)

        evicted_batch = evicted_state[0].reserved_batch
        
        # remove instance from all later timestamps
        def _remove_instance(timestamp, states):
            for i, state in enumerate(states):
                if state.model.id == instance_id:
                    states.pop(i)
                    return
            raise RuntimeError(f"Instance ID {instance_id} not found @ {timestamp}")

        eviction_end_time = start_time + evict_time
        self._insert_state_marker(eviction_end_time, _remove_instance, _remove_instance)
        return evicted_batch
    
    def evict_any(self, model_id: int, start_time: float, evict_time: float) -> tuple[str, Batch | None]:
        """Evict any copy of a specific model, prioritizing instances with no
        executing batch > instances with the latest finish time.

        Args:
            model_id: ID of model to evict (not to be confused with instance ID)
            start_time: When to start eviction
            evict_time: Duration of eviction

        Returns:
            instance_id: ID of evicted instance
            evicted_batch: If evicted instance was executing a batch, return the evicted batch
        """
        evictable_states = [s for s in self.state_at(start_time)
                            if s.model.data.id == model_id and s.state == ModelState.PLACED]
        assert(len(evictable_states) > 0)

        evicted_state = max(evictable_states, key=lambda s: np.inf if not s.reserved_batch else s.reserved_until)
        evicted_batch = evicted_state.reserved_batch
        eviction_end_time = start_time + evict_time

        # remove model from all later timestamps
        def _remove_model(timestamp, states):
            for i, state in enumerate(states):
                if state.model.id == evicted_state.model.id:
                    states.pop(i)
                    return
            assert(False)

        self._insert_state_marker(eviction_end_time, _remove_model, _remove_model)
        return evicted_state.model.id, evicted_batch

    def state_at(self, time: float) -> list[ModelState]:
        # walked in reverse rather than over a reversed copy: the history holds one
        # marker per state change and grows for the whole run, so copying it here
        # costs more than the scan itself. _insert_state_marker keeps the markers in
        # timestamp order, so the first one at or before [time] is the state in force
        for (timestamp, states) in reversed(self._model_states):
            if timestamp <= time:
                return states
        return []
    
    def does_have_copy(self, model_id: int, time: float) -> bool:
        """Returns True if any instance of given model exists, either placed or in fetch.
        """
        return any(state.model.data.id == model_id and 
                   state.state in [ModelState.IN_FETCH, ModelState.PLACED]
                   for state in self.state_at(time))
    
    def does_have_idle_copy(self, model_id: int, time: float) -> bool:
        """Returns True if exists an unreserved instance of model, either placed or in fetch.
        """
        return any(state.model.data.id == model_id and (not state.reserved_batch) and 
                   state.state in [ModelState.IN_FETCH, ModelState.PLACED]
                   for state in self.state_at(time))
    
    def reserve_idle_copy(self, model_data: ModelData, time: float, reserved_batch: Batch, reserve_for: float) -> str:
        """If there is an unreserved instance of a given model, reserve it for
        a given amount of time after the instance is active.

        Args:
            model_data: Model to reserve an instance of
            time: Time to start reservation
            reserved_batch: Batch to reserve instance for
            reserve_for: Duration of active time to reserve for. That is, if
            instance is in fetch, it will reserve for reserve_for time AFTER
            the instance finishes fetching.

        Returns:
            id: Reserved instance ID
        """
        assert(reserve_for > 0)

        idle_instances = [s for s in self.state_at(time) if s.model.data.id == model_data.id and
                          s.state in [ModelState.PLACED, ModelState.IN_FETCH] and
                          not s.reserved_batch]
        assert(len(idle_instances) > 0)

        reserved_instance = min(idle_instances, key=lambda s: s.model.active_from)
        reserve_until = max(reserved_instance.model.active_from, time) + reserve_for
        reserve_from = max(reserved_instance.model.active_from, time)

        def _occupy_one_copy(timestamp, states):
            for j, state in enumerate(states):
                if state.model.id == reserved_instance.model.id:
                    states[j].reserved_batch = reserved_batch
                    states[j].reserved_until = reserve_until
                    states[j].reserve_from = reserve_from
                    return
            assert(False)

        # reserve 1 idle copy from start to exec end
        self._insert_state_marker(time, _occupy_one_copy, _occupy_one_copy)

        return reserved_instance.model.id

    def release_busy_model(self, batch_id: int, time: float):
        """
            Releases a previously occupied/reserved copy of model that was
            executing batch specified by [batch_id] at [time].
        """
        def _release_one_copy(timestamp, states):
            for i, state in enumerate(states):
                if state.reserved_batch and state.reserved_batch.id == batch_id and \
                    state.state in [ModelState.PLACED, ModelState.IN_FETCH]:
                    states[i].reserved_batch = None
                    states[i].reserved_until = -1
                    return
            assert(False) # no batch of [batch_id] found
        self._insert_state_marker(time, _release_one_copy, _release_one_copy)

    def get_instance_state(self, instance_id, time: float):
        for state in self.state_at(time):
            if state.model.id == instance_id:
                return state
        return None
            
    def reserve_instance(self, instance_id, time: float, reserved_batch: Batch, reserve_for: float):
        reserved_instance = self.get_instance_state(instance_id, time)
        reserve_until = max(reserved_instance.model.active_from, time) + reserve_for
        reserve_from = max(reserved_instance.model.active_from, time)

        def _occupy_one_copy(timestamp, states):
            for j, state in enumerate(states):
                if state.model.id == reserved_instance.model.id:
                    assert(states[j].reserved_batch == None)
                    states[j].reserved_batch = reserved_batch
                    states[j].reserved_until = reserve_until
                    states[j].reserve_from = reserve_from
                    return
            assert(False)

        # reserve 1 idle copy from start to exec end
        self._insert_state_marker(time, _occupy_one_copy, _occupy_one_copy)