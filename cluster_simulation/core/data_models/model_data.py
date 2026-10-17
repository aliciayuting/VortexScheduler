from core.model import Model
from core.configs.workflow_config import *
import core.configs.gen_config as gcfg

from scipy.stats import linregress
from uuid import uuid4
from copy import deepcopy

import numpy as np


def lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Returns the (mu, sigma) of the underlying Normal for a log-normal
    distribution with the given mean and coefficient of variation.
    """
    sigma_sq = np.log(1 + cv**2)
    return np.log(mean) - sigma_sq / 2, np.sqrt(sigma_sq)


class ModelData:
    """
    Uninstantiated model containing model properties/data.
    """

    def __init__(self, id: int, size: float, max_batch_size: int, 
                 batch_exec_times: dict[int, dict[int, float]], exec_cvs: dict[int, float]):
        """Initialize ModelData object.

        Args:
            id: Model ID (index in MODELS from config)
            size: Model size in KB
            batch_sizes: All valid batch sizes
            batch_exec_times: Worker memory size -> batch size -> mean exec time
            exec_cvs: Worker memory size -> coefficient of variation for batch exec times
        """

        self.id = id
        self.size = size
        
        assert(max_batch_size >= 1)
        self.max_batch_size = max_batch_size

        self.batch_exec_times = {}
        for worker_size, worker_exec_times in batch_exec_times.items():
            assert(worker_size * 10**6 in gcfg.VALID_WORKER_SIZES)

            self.batch_exec_times[worker_size] = {}

            m, b = None, None
            if self.max_batch_size > 1:
                # only include data for valid batch sizes
                provided_batch_sizes = sorted([bsize for bsize in worker_exec_times.keys()
                                               if bsize <= self.max_batch_size])
                if len(provided_batch_sizes) > 1:
                    m, b, _, _, _ = linregress(
                        provided_batch_sizes, 
                        [worker_exec_times[bsize] for bsize in provided_batch_sizes])

            # NOTE/TODO: mixing regression & provided data can lead to larger batch having lower exec time
            for bsize in range(1, self.max_batch_size+1, 1):
                if bsize in worker_exec_times:
                    # use provided data if exists
                    self.batch_exec_times[worker_size][bsize] = worker_exec_times[bsize]
                else:
                    assert(m and b) # if using linreg need at least 2 samples
                    self.batch_exec_times[worker_size][bsize] = m * bsize + b
        
        assert(all(worker_size * 10**6 in gcfg.VALID_WORKER_SIZES for worker_size in exec_cvs.keys()))
        self.exec_cvs = exec_cvs

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, value):
        return isinstance(value, ModelData) and self.id == value.id

    def __str__(self):
        return f"[ModelData ID {self.id}]\n\t \
            Size: {self.size}\n\t \
            Max batch sizes: {self.max_batch_size}\n\t \
            Exec times: {self.batch_exec_times}"

    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        return ModelData(
            self.id,
            self.size,
            self.max_batch_size,
            deepcopy(self.batch_exec_times),
            deepcopy(self.exec_cvs))

    def create_instance(self, time: float, fetch_time: float) -> Model:
        return Model(uuid4(), self, time, time + fetch_time)
    
    def get_randomized_exec_time(self, batch_size: int, worker_size: int) -> float:
        """Returns the execution time of a batch on a worker of a given size,
        sampled from the distribution set by gcfg.EXEC_TIME_DISTRIBUTION with the
        configured mean and CV.

        LOGNORMAL samples are returned as drawn: the configured mean and CV are
        those of the log-normal itself, not of the underlying Normal.

        NORMAL samples are held within half and one and a half times the
        configured mean by resampling: a draw outside those bounds is discarded and
        redrawn until one falls inside, so the result is a Normal truncated at
        1 / (2 * CV) standard deviations either side of the mean. The truncation is
        symmetric, so the mean is still the configured one; the spread comes out
        slightly below the configured CV, by a negligible amount for the models
        benchmarked at a few percent and noticeably for a model whose CV
        approaches 1.

        Args:
            batch_size: Size of batch to execute
            worker_size: Memory size of worker (GB)
        """
        assert(batch_size <= self.max_batch_size)
        assert(worker_size in self.batch_exec_times)

        exact_exec_time = self.batch_exec_times[worker_size][batch_size]

        if gcfg.EXEC_TIME_DISTRIBUTION == "LOGNORMAL":
            mu, sigma = lognormal_params(exact_exec_time, self.exec_cvs[worker_size])
            return np.random.lognormal(mu, sigma)

        assert(gcfg.EXEC_TIME_DISTRIBUTION == "NORMAL")
        stddev = self.exec_cvs[worker_size] * exact_exec_time

        while True:
            sampled = np.random.normal(exact_exec_time, stddev)
            if 0.5 * exact_exec_time <= sampled <= 1.5 * exact_exec_time:
                return sampled