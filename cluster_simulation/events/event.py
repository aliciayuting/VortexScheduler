from uuid import UUID


class Agent:
    CLIENT = 0
    NETWORK = 1
    SCHEDULER = 2
    WORKER = 3
    LOGGER = 4
    VERIFIER = 5


class EventType:
    def __init__(self, id: int, name: str, kwargs: dict[str, bool],
                 emitter_types: set[int], listener_types: set[int]):
        """
        Args:
            id: Unique integer ID
            name: Human readable name for event
            kwargs: Map of event argument name -> arg is required
            emitter_types: Agents who are allowed to emit this event type
            listener_types: Agents who are allowed to listen for this event type
        """

        self.id = id
        self.name = name
        self.kwargs = kwargs
        self.emitter_types = emitter_types
        self.listener_types = listener_types

    def is_kwargs_valid(self, kwargs: dict[str, any]) -> bool:
        """Returns True if kwargs object is valid.
        Args:
            kwargs: Event argument name -> value
        
        Returns:
            is_valid: kwargs object contains all required fields and
            no unknown keys.
        """

        for k in self.kwargs.keys():
            if self.kwargs[k] and k not in kwargs:
                print(f"Missing required key {k} in kwargs")
                return False
            
        for k in kwargs.keys():
            if k not in self.kwargs:
                print(f"Unknown key {k} in kwargs")
                return False

        return True

    def can_register_emitter(self, emitter_type: int) -> bool:
        return emitter_type in self.emitter_types

    def can_register_listener(self, listener_type: int) -> bool:
        return listener_type in self.listener_types

    def __str__(self):
        return f"[Event ID: {self.id}] {self.name}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, value):
        return value != None and type(value) == EventType and value.id == self.id
    
    def __hash__(self):
        return hash(self.id)


class Event:
    def __init__(self, time: float, type: EventType, kwargs: dict[str, any]):
        self.time = time
        self.type = type

        assert(self.type.is_kwargs_valid(kwargs))
        self.kwargs = kwargs

    def __str__(self):
        return f"[{self.time}] {self.type} {self.kwargs}"
    
    def __repr__(self):
        return self.__str__()
    
    def __lt__(self, other):
        return (self.time, self.type.id) < (other.time, other.type.id)
    

class EventListener:
    def __init__(self, listener_type: int):
        self.listener_type = listener_type

    def set_id(self, id: UUID):
        self.listener_id = id

    def on_event(self, event: Event):
        raise NotImplementedError()