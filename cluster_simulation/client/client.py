from events.event_manager import EventManager
from events.event import *
from events.event_types import *

from core.data_models.workflow import Workflow

from uuid import UUID

import core.configs.gen_config as gcfg
import numpy as np


class Client(EventListener):

    def __init__(self, id: UUID, em: EventManager):
        super().__init__(Agent.CLIENT)

        self.id = id
        self.em = em

        self.em.register_listener(self, {
            EVENT_TYPES[EventIds.RESPONSE_RECEIVED_AT_CLIENT],
            EVENT_TYPES[EventIds.JOBS_DROPPED]
        })

        self.emitter_id = self.em.register_emitter(Agent.CLIENT, {
            EVENT_TYPES[EventIds.JOB_SENT_TO_SCHEDULER]
        })

        # job ID: (create time, received response time, did finish job, deadline, job)
        self.jobs: dict[int, (float, float, bool, float, Job)] = {}

    def on_event(self, event: Event):
        if event.type.id == EventIds.RESPONSE_RECEIVED_AT_CLIENT:
            if event.kwargs["client_id"] != self.id:
                return

            # should not have logged before
            assert(self.jobs[event.kwargs["job"].id][1] == -1)

            self.jobs[event.kwargs["job"].id] = (
                self.jobs[event.kwargs["job"].id][0],
                event.time,
                True,
                self.jobs[event.kwargs["job"].id][3],
                self.jobs[event.kwargs["job"].id][4]
            )

            if gcfg.ENABLE_CONSOLE_PRINT:
                print(f"Remaining jobs for client {self.id}: {len(self.jobs.keys()) - len([v for v in self.jobs.values() if v[2]])}")
        
        elif event.type.id == EventIds.JOBS_DROPPED:
            for job_id in event.kwargs["job_ids"]:
                if job_id in self.jobs:
                    # workers holding parallel stages of the same job may drop it
                    # simultaneously; keep the first drop and never drop a job
                    # that already completed
                    if self.jobs[job_id][1] != -1:
                        assert(not self.jobs[job_id][2])
                        continue

                    self.jobs[job_id] = (
                        self.jobs[job_id][0],
                        event.time,
                        self.jobs[job_id][2],
                        self.jobs[job_id][3],
                        self.jobs[job_id][4]
                    )

        else:
            raise ValueError(f"Client received unregistered event: {event}")
    
    def generate_jobs(self, workflow: Workflow, num_jobs: int, start_time: float, send_rate: float, slo: float, job_id_range_start: int) -> float:
        """Generates a given number of jobs at random intervals according to the configured
        workload distribution.

        Args:
            workflow: Workflow to generate jobs from
            num_jobs: Total number of jobs to generate
            start_time: Time to start sending jobs (ms)
            send_rate: Desired average send rate (qps)

        Returns:
            last_create_time: Create time of the last job generated
        """

        prev_time = start_time
        last_create = -1
        for n in range(num_jobs):
            job_create_delay = 1 / send_rate * 1000

            if gcfg.WORKLOAD_DISTRIBUTION == "POISSON":
                job_create_delay = np.random.poisson(lam=(1 / send_rate * 1000))
            elif gcfg.WORKLOAD_DISTRIBUTION == "GAMMA":
                shape = (1 / send_rate * 1000)**2 / gcfg.GAMMA_CV**2
                scale = gcfg.GAMMA_CV**2 / (1 / send_rate * 1000)
                job_create_delay = np.random.gamma(shape, scale)
            
            job_create_time = prev_time + job_create_delay
            job = Job(created_at=job_create_time,
                      workflow=workflow,
                      job_id=job_id_range_start + n, 
                      client_id=self.id,
                      slo=slo)
            
            last_create = job_create_time
            prev_time = job_create_time
            self.jobs[job_id_range_start + n] = (
                job_create_time, -1, False, job_create_time + slo, job)

            self.em.add_event(
                Event(job_create_time,
                      EVENT_TYPES[EventIds.JOB_SENT_TO_SCHEDULER],
                      kwargs={"job": job, "from_client_id": self.id, "ignore_transfer_time": not gcfg.ENABLE_NETWORKING_DELAYS}),
                self.emitter_id)

        return last_create