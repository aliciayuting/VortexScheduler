import sys
import json


def print_arg(key, config, default, add_quotes=False):
    if add_quotes:
        print(f"\"{config[key] if key in config else default}\"")
    else:
        print(config[key] if key in config else default)


if __name__ == "__main__":
    configs_path = sys.argv[1]
    configs = json.loads(open(configs_path, "r").read())

    for config in configs["CONFIGS"]:
        formatted_client_configs = [{ int(jt): cfg for jt, cfg in cc.items() } for cc in config["CLIENT_CONFIGS"]]
        formatted_alloc = [ (int(k), v) for mig in config["CUSTOM_ALLOCATION"] for k, v in mig.items() ]

        print(config["SCHEDULER_TYPE"])
        print(config["OUT_PATH"])
        print(config["TOTAL_NUM_OF_NODES"])
        print(formatted_client_configs)
        print_arg("WORKLOAD_DISTRIBUTION", config, "POISSON", True)
        print_arg("GAMMA_CV", config, 10)
        print_arg("LOAD_INFORMATION_STALENESS", config, 1)
        print_arg("PLACEMENT_INFORMATION_STALENESS", config, 1)
        print_arg("RESCHEDULE_THREASHOLD", config, 1.5)
        print_arg("FLEX_LAMBDA", config, 3.03)
        print_arg("HERD_K", config, 1.3)
        print_arg("HERD_PERIODICITY", config, 0)
        print(config["SLO_SLACK"])
        print_arg("SLO_GRANULARITY", config, "JOB", True)
        print(config["ENABLE_MULTITHREADING"])
        print(config["ENABLE_MODEL_PREFETCH"])
        print(config["ENABLE_DYNAMIC_MODEL_LOADING"])
        print(config["ALLOCATION_STRATEGY"])
        print(formatted_alloc)
        print()