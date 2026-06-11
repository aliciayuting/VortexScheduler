import pandas as pd
import core.configs.gen_config as gcfg

from queue import PriorityQueue

from events.event_types import *
from events.event import *
from uuid import UUID, uuid4


class EventManager:

    def __init__(self):
        self._event_queue = PriorityQueue()
        self._prev_time: float = -1

        self._registered_listeners: dict[UUID, EventListener] = {}

        # event ID -> emitter/listener IDs authorized to emit/listen for event
        self._event_emitters: dict[int, list[UUID]] = {}
        self._event_listeners: dict[int, list[UUID]] = {}

        # (model_id, worker_id, time) keys for pending CHECK_QUEUE events
        self._pending_check_queue: set[tuple] = set()

        # (time, type_id, str(kwargs)) keys for all other pending events
        self._pending_events: set[tuple] = set()

        self._event_log_rows: list = []


    def register_emitter(self, emitter_type: int, event_types: set[EventType]) -> UUID:
        """Registers new event emitter.

        Args:
            emitter_type: Agent type of emitter
            event_types: Set of all events emitter may emit

        Returns:
            emitter_id: Unique ID for emitter
        """
        assert(all(et.id in EVENT_TYPES for et in event_types))
        assert(all(et.can_register_emitter(emitter_type) for et in event_types))

        emitter_id = uuid4()
        for et in event_types:
            if et.id not in self._event_emitters:
                self._event_emitters[et.id] = []
            self._event_emitters[et.id].append(emitter_id)

        return emitter_id
    

    def register_listener(self, listener: EventListener, event_types: set[EventType]):
        """Registers new event listener.

        Args:
            listener_type: Agent type of listener
            event_types: Set of all events listener may emit
        """
        assert(all(et.id in EVENT_TYPES for et in event_types))
        assert(all(et.can_register_listener(listener.listener_type) for et in event_types))

        listener_id = uuid4()
        self._registered_listeners[listener_id] = listener
        listener.set_id(listener_id)
        
        for et in event_types:
            if et.id not in self._event_listeners:
                self._event_listeners[et.id] = []
            self._event_listeners[et.id].append(listener_id)
    

    def add_event(self, event: Event, emitter_id: UUID):
        """Enqueues event onto event queue.
        """
        assert(emitter_id in self._event_emitters[event.type.id])

        if event.type.id == EventIds.CHECK_QUEUE_AT_WORKER:
            key = (event.kwargs["model_id"], event.kwargs["worker_id"], event.time)
            if key in self._pending_check_queue:
                print("[WARNING] Duplicate CHECK_QUEUE event queued, ignoring add_event")
                return
            self._pending_check_queue.add(key)
        elif gcfg.ENABLE_DUPLICATE_EVENT_CHECK:
            key = (event.time, event.type.id, str(event.kwargs))
            if key in self._pending_events:
                raise RuntimeError(f"Duplicate event detected: {event}")
            self._pending_events.add(key)

        self._event_queue.put(event)


    def has_events(self) -> bool:
        """Returns True if there is at least one event awaiting processing.
        """
        return self._event_queue.qsize() > 0

    
    def process_next_event(self):
        """Dequeues the next event from the event queue and notifies all
        registered listeners.
        """

        event: Event = self._event_queue.get()
        assert(event.time >= self._prev_time)

        if event.type.id == EventIds.CHECK_QUEUE_AT_WORKER:
            self._pending_check_queue.discard(
                (event.kwargs["model_id"], event.kwargs["worker_id"], event.time))
        elif gcfg.ENABLE_DUPLICATE_EVENT_CHECK:
            self._pending_events.discard(
                (event.time, event.type.id, str(event.kwargs)))

        if gcfg.PRODUCE_EVENT_LOG:
            self._event_log_rows.append([event.time, str(event)])

        # print("Received event: ", event)
        # print()

        # notify all listeners
        for listener_id in self._event_listeners[event.type.id]:
            self._registered_listeners[listener_id].on_event(event)

        self._prev_time = event.time

    @property
    def event_log(self):
        return pd.DataFrame(self._event_log_rows, columns=["time", "event"])
