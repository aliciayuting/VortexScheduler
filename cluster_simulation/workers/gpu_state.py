import copy

from core.config import *
from core.model import Model
from core.task import Task
from core.batch import Batch


class ModelState:
    """
        Wrapper class representing a Model placed on a GPU.
    """
    
    PLACED = 0
    PRE_FETCH = 1 # reserved for a model that will be fetched
    IN_FETCH = 2
    IN_EVICT = 3

    def __init__(self, model: Model, state: int, reserved_batch: Batch=None, reserved_until=-1, size=0):
        self.model = model
        self.size = size if size > 0 else model.model_size
        self.state = state
        self.reserved_batch = reserved_batch
        self.reserved_until = reserved_until

    def __eq__(self, value):
        return type(value) == ModelState and \
            self.model.model_id == value.model.model_id and \
            self.state == value.state and \
            self.reserved_batch == value.reserved_batch
    
    def __str__(self):
        executing_batch = f'EXECUTING BATCH {self.reserved_batch.id}' if self.reserved_batch else 'NOT IN USE'
        return f"<[{self._state_to_str()}] [{executing_batch}] Model ID: {self.model.model_id if self.model else -1}>"
    
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

    def can_fetch_model(self, model: Model, time: float) -> bool:
        """
            Returns True if a new copy of [model] can be fetched to the
            GPU as is with no evictions.
        """
        return self.available_memory(time) >= model.model_size
    
    def can_fetch_model_on_eviction(self, model: Model, time: float) -> bool:
        """
            Return True if a new copy of [model] can be fetched to the GPU
            upon evicting some number of placed models not in use.
        """
        # cannot use space occupied by models currently being fetched/evicted or used
        return (self.available_memory(time) + \
                sum(state.size for state in self.state_at(time)
                    if state.state == ModelState.PLACED and not state.reserved_batch)) >= model.model_size

    def prefetch_model(self, model: Model):
        """
            Preload [model] onto GPU for time 0. Ignores fetching cost.
        """
        assert(self.can_fetch_model(model, 0))

        if len(self._model_states) == 0:
            self._model_states.append((0, [ModelState(model, ModelState.PLACED)]))
        else:
            self._model_states[0][1].append(ModelState(model, ModelState.PLACED))
    
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

    def fetch_model(self, model: Model, start_time: float, fetch_time: float, reserved_batch: Batch=None, reserve_until=-1):
        """
            Fetches a new copy of [model] to the GPU if there is enough available
            memory without additional evictions.

            If [reserved_batch] and [reserve_until] are specified, reserves the
            model for execution of [reserved_batch] until time [reserve_until].
        """
        assert(model != None)
        assert(self.can_fetch_model(model, start_time))

        fetch_end_time = start_time + fetch_time

        if len(self._model_states) == 0:
            # mark when fetch begins and ends
            self._model_states.append((start_time, [ModelState(model, ModelState.IN_FETCH,
                                                               reserved_batch=reserved_batch,
                                                               reserved_until=reserve_until)]))
            self._model_states.append((fetch_end_time, [ModelState(model, ModelState.PLACED, 
                                                                   reserved_batch=reserved_batch,
                                                                   reserved_until=reserve_until)]))
            return
        
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

    
    def evict_model(self, model: Model, start_time: float, evict_time: float, reserve_until=-1):
        """
            Evicts [model] starting at [start_time] in [evict_time] time.
            Reserves evicted space until [reserve_until]. This prevents other models
            from being loaded in space that may be intended to fetch a specific model.
            Does not reserve if [reserve_until] < 0.
        """
        assert(model in self.placed_models(start_time))

        eviction_end_time = start_time + evict_time

        # remove model from all later timestamps
        def _remove_model(timestamp, states):
            for state in states:
                if state.state == ModelState.PLACED and state.model == model:
                    states.remove(state)
                    return
            assert(False)

        self._insert_state_marker(eviction_end_time, _remove_model, _remove_model)

        # def _begin_model_eviction(timestamp, states):
        #     for state in states:
        #         if state.state == ModelState.PLACED and state.model == model and not state.is_reserved_for_batch:
        #             state.state = ModelState.IN_EVICT
        #             return
        #     assert(False) # should not happen: no model exists to evict

        # add eviction start marker
        # self._insert_state_marker(start_time, _begin_model_eviction,
        #                           lambda t, states: _begin_model_eviction(t, states) if t < eviction_end_time else None)
        
        # if reserve_until >= 0:
        #     self.reserve_model_space(model, model.model_size, eviction_end_time, reserve_until)
        
    def reserve_model_space(self, model: Model, size: float, start_time: float, end_time: float):
        """
            Reserves [size] extra space for [model] from [start_time] to [end_time].
            Used during evictions when additional space must be reserved in addition
            to space from evicted or currently evicting models for when [model] is
            fetched. Prevents other models from being fetched in space made for
            [model].
        """
        assert(size > 0)

        # mark reservation start
        self._insert_state_marker(start_time,
                                  lambda _, states: states.append(ModelState(model, ModelState.PRE_FETCH, size=size)),
                                  lambda t, states: states.append(ModelState(model, ModelState.PRE_FETCH, size=size)) if t < end_time else None)
        
        def _remove_reservation(timestamp, states):
            for state in states:
                if state.model == model and state.state == ModelState.PRE_FETCH and state.size == size:
                    states.remove(state)
                    return

        # mark reservation end
        self._insert_state_marker(end_time, _remove_reservation, lambda _, states: None)

    def state_at(self, time: float) -> list[ModelState]:
        for (timestamp, states) in self._model_states[::-1]:
            if timestamp <= time:
                return states
        return []

    def placed_models(self, time: float) -> list[Model]:
        return [state.model for state in self.state_at(time) if state.state == ModelState.PLACED]
    
    def placed_model_states(self, time: float) -> list[ModelState]:
        states = self.state_at(time)
        if len(states) == 0:
            return []
        return [state for state in states if state.state == ModelState.PLACED]
    
    def does_have_idle_copy(self, model: Model, time: float) -> bool:
        return any(state.model.model_id == model.model_id and not state.reserved_batch and state.state in [ModelState.IN_FETCH, ModelState.PLACED]
                   for state in self.state_at(time))
    
    def reserve_idle_copy(self, model: Model, time: float, reserved_batch: Batch, reserve_until: float):
        """
            If there is an idle copy of [model], reserve it to execute a batch
            starting from [time]. When execution finishes, a call to
            [release_busy_copy] is required.
        """
        assert(self.does_have_idle_copy(model, time))
        assert(reserve_until > time)

        def _occupy_one_copy(timestamp, states):
            for j, state in enumerate(states):
                if state.model.model_id == model.model_id and \
                    state.state in [ModelState.PLACED, ModelState.IN_FETCH] and \
                    not state.reserved_batch:
                    states[j].reserved_batch = reserved_batch
                    states[j].reserved_until = reserve_until
                    return
            assert(False) # should not reach! (no idle copies)

        # reserve 1 idle copy from start to exec end
        self._insert_state_marker(time, _occupy_one_copy, _occupy_one_copy)

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

    def shortest_time_to_fetch_end(self, model_id: int, time: float) -> float:
        """
            If at least one copy of model with [model_id] is currently being fetched,
            returns the time between [time] and the end of the fetch for the copy
            which will finish fetching earliest.
        """
        current_fetching_count = sum(s.model.model_id==model_id and s.state==ModelState.IN_FETCH 
                                     for s in self.state_at(time))
        assert(current_fetching_count > 0)
        for (timestamp, states) in self._model_states:
            if timestamp >= time:
                fetching_count = sum(s.model.model_id==model_id and s.state==ModelState.IN_FETCH 
                                     for s in states)
                if fetching_count < current_fetching_count:
                    return timestamp - time
        return -1