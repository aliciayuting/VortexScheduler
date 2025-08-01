from collections import deque
import csv
import enum
from hmac import new
import json
import logging
import argparse
import random

import math
from performance import get_performance_metrics


from scheuler.dynamic_scheduler import DynamicScheduler
from scheuler.simple_scheduler import SimpleScheduler 
from utils import SortedQueue
from vary_trace import *


batch_runtimes = {}

def setup_logging(log_level_str="INFO", scheduler_name="", preemption=False, trace_variation="", slo_factor=None, max_batch_size=None, output_file=None):
    """Set up logging with specified level"""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = level_map.get(log_level_str.upper(), logging.INFO)
    
    # Create log filename based on scheduler name, preemption, trace variation, slo factor, and max batch size
    if output_file:
        # If output_file is provided, use it as the base name
        log_filename = f'./output/{output_file}.log'
    else:
        # Use the current naming convention
        preemption_suffix = "-preemption" if preemption else ""
        trace_suffix = f"-{trace_variation}" if trace_variation else ""
        slo_suffix = f"-slo-{slo_factor}" if slo_factor is not None else ""
        batch_suffix = f"-batch-{max_batch_size}" if max_batch_size is not None else ""
        log_filename = f'./output/{scheduler_name}{preemption_suffix}{trace_suffix}{slo_suffix}{batch_suffix}.log'
    
    logging.basicConfig(
        level=log_level,
        format='%(filename)20s:%(lineno)4d - %(levelname)8s - %(message)s',
        handlers=[
                logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class Event(enum.Enum):
    CHECK_PREEMPTION = "check preemption"
    BATCH_FINISHED = "batch finished"
    NEW_REQS_ARRIVED = "new reqs arrived"

class Request:
    arrival_time: int
    id: int
    queue_time: float
    execution_time: float
    finish_time: float
    batch_size: int
    dropped_time: float
    slo_factor: float
    deadline: float

    def __init__(self, arrival_time: int, id: int, slo_factor: float):
        self.arrival_time = arrival_time
        self.id = id
        self.queue_time = None
        self.dropped_time = None
        self.execution_time = None
        self.finish_time = None
        self.batch_size = None
        self.slo_factor = slo_factor
        self.deadline = self.arrival_time + self.slo_factor * batch_runtimes[1]

    def __str__(self):
        return f"Request(id={self.id}, arrival_time={self.arrival_time}, slo_factor={self.slo_factor}, deadline={self.deadline})" if self.queue_time is None else f"Request(id={self.id}, arrival_time={self.arrival_time}, slo_factor={self.slo_factor}, deadline={self.deadline}, queue_time={self.queue_time}, batch_size={self.batch_size}, execution_time={self.execution_time}, finish_time={self.finish_time}, dropped_time={self.dropped_time})"

    def schedule(self, current_time: float, batch_size: int, batch_runtime: float):
        self.queue_time = current_time - self.arrival_time
        self.batch_size = batch_size
        self.execution_time = batch_runtime
        self.finish_time = current_time + batch_runtime
    
    def preempt(self):
        self.queue_time = None
        self.batch_size = None
        self.execution_time = None
        self.finish_time = None

    def get_dropped(self, current_time: float):
        self.dropped_time = current_time

    def __repr__(self):
        return self.__str__()



queue = SortedQueue()
future_requests = SortedQueue()
finished_requests = []


current_batch = SortedQueue()
current_time = float (0)
batch_finish_time = math.inf





def fetch_new_requests(current_time: int, future_requests: SortedQueue, queue: SortedQueue):
    
    new_reqs = []
    while len(future_requests) > 0 and future_requests[0].arrival_time <= current_time:
        req = future_requests.pop()
        queue.append(req)
        new_reqs.append(req.id)
    return new_reqs

def drop_requests(current_time: int, queue: SortedQueue, finished_requests: list[Request]):
    dropped_reqs = []
    while len(queue) > 0 and queue[0].deadline < current_time + batch_runtimes[1]:
        req = queue.pop()
        req.get_dropped(current_time)
        finished_requests.append(req)
        dropped_reqs.append(req.id)

    return dropped_reqs


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Scheduler simulation with configurable logging')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level (default: INFO)')
    parser.add_argument('--scheduler', required=True, choices=['simple', 'dynamic'],
                       help='Scheduler type: simple or dynamic')
    parser.add_argument('--preemption', action='store_true',
                       help='Enable preemption (default: False)')
    parser.add_argument('--vary-trace', choices=['compress', 'multi-user'], default='compress',
                       help='Trace variation: compress or multi-user (default: compress)')
    parser.add_argument('--compress-ratio', type=float, default=0.3,
                       help='Compression ratio for trace compression (default: 0.3)')
    parser.add_argument('--num-user', type=int, default=10,
                       help='Number of users for multi-user trace (default: 10)')
    parser.add_argument('--slo-factor', type=float, default=5.0,
                       help='SLO factor multiplier for base latency (default: 5.0)')
    parser.add_argument('--max-batch-size', type=int, default=16,
                       help='Maximum batch size for scheduling (default: 16)')
    parser.add_argument('--offline-num-reqs', type=int, default=0,
                    help='Number of requests to simulate in the offline setting (default: 0)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Custom output filename for logging (default: scheduler_name-preemption-trace_variation-slo_factor-max_batch_size.log)')
    parser.add_argument('--slo-csv', type=str, default=None,
                       help='Path to CSV file containing SLO factors and arrival times (e.g., uniform-slo.csv)')
    args = parser.parse_args()
    
    
    # Validate scheduler name

    if args.scheduler not in ['simple', 'dynamic']:
        print(f"Error: Invalid scheduler name '{args.scheduler}'. Must be 'simple' or 'dynamic'.")
        exit(1)
    
    # Validate trace variation arguments
    if args.vary_trace == 'compress' and args.compress_ratio <= 0:
        print(f"Error: compress_ratio must be positive, got {args.compress_ratio}")
        exit(1)
    if args.vary_trace == 'multi-user' and args.num_user <= 0:
        print(f"Error: num_user must be positive, got {args.num_user}")
        exit(1)
    if args.slo_factor <= 0:
        print(f"Error: slo_factor must be positive, got {args.slo_factor}")
        exit(1)
    if args.max_batch_size <= 0:
        print(f"Error: max_batch_size must be positive, got {args.max_batch_size}")
        exit(1)
    
    # Create trace variation string for file naming
    trace_variation = ""
    if args.vary_trace == 'compress':
        trace_variation = f"compress-{args.compress_ratio}"
    elif args.vary_trace == 'multi-user':
        trace_variation = f"multi-user-{args.num_user}"
    
    # Set up logging with scheduler name, preemption info, trace variation, slo factor, and max batch size
    logger = setup_logging(args.log_level, args.scheduler, args.preemption, trace_variation, args.slo_factor, args.max_batch_size, args.output_file)
    logger.info(f"Starting simulation with scheduler: {args.scheduler}, preemption: {args.preemption}, trace_variation: {trace_variation}, slo_factor: {args.slo_factor}, max_batch_size: {args.max_batch_size}, log level: {args.log_level}")

    
    # read throughput profile
    with open('./runtimes_by_batch_size.csv', mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            batch_size = int(row['bsize'])
            runtime = float(row['mean_runtime_ms'])
            batch_runtimes[batch_size] = runtime
    # print(f"Batch runtimes: {batch_runtimes}")
    
    # Read trace file
    trace_file_path = '../../workflow/azuretrace/llm_az_processed_trace.csv'
    if args.vary_trace == 'compress':
        arrival_times = generate_trace_with_simple_compression(trace_file_path, args.compress_ratio)
        logger.info(f"Generated compressed trace with ratio: {args.compress_ratio}")
    elif args.vary_trace == 'multi-user':
        arrival_times = generate_trace_with_multiple_concurrent_users(trace_file_path, args.num_user)
        logger.info(f"Generated multi-user trace with {args.num_user} users")
    else:
        logger.error(f"Invalid trace variation: {args.vary_trace}")
        exit(1)
    
    # Create requests from arrival times
    if args.slo_csv:
        # Read SLO factors and arrival times from CSV file
        logger.info(f"Reading SLO factors and arrival times from {args.slo_csv}")
        with open(args.slo_csv, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                request_id = int(row['request_id'])
                slo_factor = float(row['slo_factor'])
                arrival_time = float(row['arrival_time'])
                
                request = Request(arrival_time if args.offline_num_reqs == 0 else 0, request_id, slo_factor)
                future_requests.append(request)
                
                if args.offline_num_reqs > 0 and request_id >= args.offline_num_reqs - 1:
                    break
    else:
        # Use original method with random SLO factors
        for i, arrival_time in enumerate(arrival_times):
            # random sample a slo factor between 1.5 and 25
            # slo_factor = random.uniform(1.5, 25)
            slo_factor = args.slo_factor
            
            request = Request(arrival_time if args.offline_num_reqs == 0 else 0, i, slo_factor)
            future_requests.append(request)

            if args.offline_num_reqs > 0 and i >= args.offline_num_reqs - 1:
                break
    
    logger.info(f"Loaded {len(future_requests)} requests from the trace file")



    # slo_factor = args.slo_factor
    # base_latency = batch_runtimes[1]
    # slo = slo_factor * base_latency

    # Initialize scheduler based on command line arguments
    if args.scheduler == 'simple':
        scheduler = SimpleScheduler(max_batch_size=args.max_batch_size, batch_runtimes=batch_runtimes, logger=logger)
    elif args.scheduler == 'dynamic':
        scheduler = DynamicScheduler(max_batch_size=args.max_batch_size, batch_runtimes=batch_runtimes, logger=logger)
    else:
        logger.error(f"Invalid scheduler name: {args.scheduler}")
        exit(1)

    # Prepare JSON file for writing finished requests
    if args.output_file:
        # If output_file is provided, use it as the base name
        json_filename = f'output/{args.output_file}_finished_reqs.json'
    else:
        # Use the current naming convention
        preemption_suffix = "-preemption" if args.preemption else ""
        trace_suffix = f"-{trace_variation}" if trace_variation else ""
        slo_suffix = f"-slo-{args.slo_factor}"
        batch_suffix = f"-batch-{args.max_batch_size}"
        json_filename = f'output/{args.scheduler}{preemption_suffix}{trace_suffix}{slo_suffix}{batch_suffix}_finished_reqs.json'
    current_time = float(future_requests[0].arrival_time)

    num_iters = 0;


    if args.offline_num_reqs > 0:
        fetch_new_requests(current_time, future_requests, queue)
        scheduler.offline_schedule(current_batch, queue, current_time, finished_requests)
    else:
        while (True):

            # check the event
            if len(current_batch) == 0:
                event = Event.NEW_REQS_ARRIVED
            elif current_time == batch_finish_time:
                event = Event.BATCH_FINISHED
            else:
                event = Event.CHECK_PREEMPTION

            logger.info("\n" + "-"*10  +  f"Current time: {current_time}" + f" ({event.value})" + "-"*10)

            # drop requests
            drop_reqs = drop_requests(current_time, queue, finished_requests)
            logger.info(f"[Dropped requests] {drop_reqs}")
            
            # fetech new reqs
            new_reqs = fetch_new_requests(current_time, future_requests, queue)
            if len(new_reqs) > 0:
                logger.info(f"[New requests] {new_reqs}")

            logger.info(f"[Current batch] {[req.id for req in current_batch]}")
            logger.info(f"[Queue] {[req.id for req in queue]}")
            # logger.info(f"[Queue] {queue.requests}")

            # check if we need to do preemption
            if (event == Event.CHECK_PREEMPTION and args.preemption):
                # do something
                logger.info(f"[Check preemption] at {current_time}")
                do_preemption = scheduler.preempt(current_batch, queue, current_time, batch_finish_time)
                if do_preemption:
                    logger.info(f"[New scheduled batch] {[req.id for req in current_batch]}")
                    duration = batch_runtimes[len(current_batch)]
                    batch_finish_time = current_time + duration


            # check if the current batch is finished
            if event == Event.BATCH_FINISHED:
                assert len(current_batch) > 0, f"The current batch should be finished at this time but the current batch is empty."
                req_ids = []
                
                while len(current_batch) > 0:
                    req = current_batch.pop()
                    assert req.finish_time == current_time, f"The finish time of the request {req.id} is not correct."
                    req_ids.append(req.id)
                    finished_requests.append(req)
                logger.info(f"[Batch finished] {req_ids}")
                batch_finish_time = math.inf

            # schedule the next batch
            if event != Event.CHECK_PREEMPTION:
                next_check_time, _ = scheduler.schedule(current_batch, queue, current_time)
                if len(current_batch) > 0:
                    logger.info(f"[Scheduled] {[req.id for req in current_batch]}")
                    duration = batch_runtimes[len(current_batch)]
                    batch_finish_time = current_time + duration
                    # update the timestep in the current batch
                    for req in current_batch:
                        req.schedule(current_time, len(current_batch), batch_runtimes[len(current_batch)])
                else:
                    logger.info(f"[No batch scheduled] queue length: {len(queue)}")

            # check if the trace is finished
            if len(future_requests) == 0 and len(queue) == 0 and len(current_batch) == 0:
                logger.info("Trace finished")
                break
            
            # check what is the next event
            next_req_arrival_time = future_requests[0].arrival_time if len(future_requests) > 0 else math.inf
            assert not (batch_finish_time==math.inf and next_check_time == math.inf and next_req_arrival_time == math.inf), f"The 3 times are all inf, which is not possible."
            current_time = min(next_check_time, batch_finish_time, next_req_arrival_time)
            logger.info(f"[Time] batch finish time: {batch_finish_time} next req arrival time: {next_req_arrival_time} next check time: {next_check_time}")


            num_iters += 1


            # if num_iters > 20:
            #     break

    
    # Write finished requests to JSON file
    finished_requests_data = [req.__dict__ for req in finished_requests]
    with open(json_filename, 'w') as jsonfile:
        json.dump(finished_requests_data, jsonfile, indent=2)

    logger.info(f"Finished requests written to {json_filename}")
    



    # Create a simple logger for performance metrics (message-only format)
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    
    # Create a handler that writes to the same file as the main logger
    if args.output_file:
        # If output_file is provided, use it as the base name
        perf_log_filename = f'./output/{args.output_file}.log'
    else:
        # Use the current naming convention
        preemption_suffix = "-preemption" if args.preemption else ""
        trace_suffix = f"-{trace_variation}" if trace_variation else ""
        slo_suffix = f"-slo-{args.slo_factor}"
        batch_suffix = f"-batch-{args.max_batch_size}"
        perf_log_filename = f'./output/{args.scheduler}{preemption_suffix}{trace_suffix}{slo_suffix}{batch_suffix}.log'
    perf_handler = logging.FileHandler(perf_log_filename)
    perf_handler.setLevel(logging.INFO)
    
    # Create a formatter that only shows the message
    perf_formatter = logging.Formatter('%(message)s')
    perf_handler.setFormatter(perf_formatter)
    
    # Add the handler to the logger
    perf_logger.addHandler(perf_handler)
    
    get_performance_metrics(finished_requests_data, perf_logger)






    
