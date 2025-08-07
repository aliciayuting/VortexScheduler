from core.job import *
from core.config import *
from core.network import *
from core.config import *

import numpy as np


class ExternalClient(object):
    """
        Represents a client sending requests of various job types.
    """
    def __init__(self, simulation, id: int, config: dict):
        self.id = id
        self.simulation = simulation
        self.job_types = config.keys()
        self.per_job_config = config

    def create_job(self, job_type: int, job_id: int, send_rate: int, current_time: float) -> Job:
        assert(job_type in self.job_types)

        job_create_delay = 1 / send_rate * 1000

        if WORKLOAD_DISTRIBUTION == "POISSON":
            job_create_delay = np.random.poisson(lam=(1 / send_rate * 1000))
        elif WORKLOAD_DISTRIBUTION == "GAMMA":
            shape = (1 / send_rate * 1000)**2 / GAMMA_CV**2
            scale = GAMMA_CV**2 / (1 / send_rate * 1000)
            job_create_delay = np.random.gamma(shape, scale)
        
        job = Job(create_time=current_time + job_create_delay,
                  job_type_id=job_type, job_id=job_id, client_id=self.id,
                  slo=self.per_job_config[job_type]["SLO"] if "SLO" in self.per_job_config[job_type] else np.inf)
        job = self.log_job_creation_time(job, current_time + job_create_delay)
        return job

    def select_initial_worker_id(self) -> int:
        initial_worker_id = np.random.choice(range(len(self.simulation.workers)))
        return initial_worker_id

    def log_job_creation_time(self, job: Job, creation_time: float) -> Job:
        for task in job.tasks:
            task.log.job_creation_timestamp = creation_time
        return job