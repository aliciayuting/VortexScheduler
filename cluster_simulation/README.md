# ML pipeline scheduler and object placement simulator

This repository contains the simulation code to run the Vortex ML pipeline scheduler on customized workflow and cluster.

##  Run simulation
### 1. set environment variable at the terminal 
#### * For Unix/Linux
`source set_env.sh`
#### * For Windows
set PYTHONPATH=%PYTHONPATH%;C:\path\to\directory\

### 2. Simulation

#### 2.1 Set Config and Workflow
User can set customized system configurations in /core/configs/.

Trace-backed workloads can use a path relative to the repository root, so the
configuration remains portable across machines:

```python
{"WORKLOAD": {
    "ARRIVAL_PROCESS": "ALITRACE",
    "TRACE_FILE_PATH": "workflow/azuretrace/llm_az_processed_trace.csv",
}}
```

Relative `TRACE_FILE_PATH` values are resolved from the repository root, not
from the current shell directory. If `TRACE_FILE_PATH` is omitted for an
`ALITRACE` workload, it defaults to the path shown above. Absolute paths remain
supported when a trace is stored outside the repository.

#### 2.2 Run simulation
Run the python file, /experiments/run_experiment.py, to start the simulation and generate log files. Defaults to `{CURRENT_DIR}/results` if path to output directory `-o` is not specified.

``` python run_experiment.py -t {centralheft,decentralheft,hashtask,shepherd} [-o OUT] ```
