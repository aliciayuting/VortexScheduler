#!/bin/zsh

configs_path=$1
echo "Parsing configs from $configs_path"

root_dir=$(pwd)
configs_out=$(python3 parse_configs.py $configs_path)

run_experiment() {
    local cfg_prop_list=("$@")

    sed -i '.bak' -e "s/^TOTAL_NUM_OF_NODES = .*$/TOTAL_NUM_OF_NODES = ${cfg_prop_list[3]}/" \
        -e "s/^CLIENT_CONFIGS = .*$/CLIENT_CONFIGS = ${cfg_prop_list[4]}/" \
        -e "s/^WORKLOAD_DISTRIBUTION = .*$/WORKLOAD_DISTRIBUTION = ${cfg_prop_list[5]}/" \
        -e "s/^GAMMA_CV = .*$/GAMMA_CV = ${cfg_prop_list[6]}/" \
        -e "s/^LOAD_INFORMATION_STALENESS = .*$/LOAD_INFORMATION_STALENESS = ${cfg_prop_list[7]}/" \
        -e "s/^PLACEMENT_INFORMATION_STALENESS = .*$/PLACEMENT_INFORMATION_STALENESS = ${cfg_prop_list[8]}/" \
        -e "s/^RESCHEDULE_THREASHOLD = .*$/RESCHEDULE_THREASHOLD = ${cfg_prop_list[9]}/" \
        -e "s/^FLEX_LAMBDA = .*$/FLEX_LAMBDA = ${cfg_prop_list[10]}/" \
        -e "s/^HERD_K = .*$/HERD_K = ${cfg_prop_list[11]}/" \
        -e "s/^HERD_PERIODICITY = .*$/HERD_PERIODICITY = ${cfg_prop_list[12]}/" \
        -e "s/^SHEPHERD_BATCHING_POLICY = .*$/SHEPHERD_BATCHING_POLICY = ${cfg_prop_list[13]}/" \
        -e "s/^SLO_SLACK = .*$/SLO_SLACK = ${cfg_prop_list[14]}/" \
        -e "s/^SLO_GRANULARITY = .*$/SLO_GRANULARITY = ${cfg_prop_list[15]}/" \
        -e "s/^ENABLE_MULTITHREADING = .*$/ENABLE_MULTITHREADING = ${cfg_prop_list[16]}/" \
        -e "s/^ENABLE_MODEL_PREFETCH = .*$/ENABLE_MODEL_PREFETCH = ${cfg_prop_list[17]}/" \
        -e "s/^ENABLE_DYNAMIC_MODEL_LOADING = .*$/ENABLE_DYNAMIC_MODEL_LOADING = ${cfg_prop_list[18]}/" \
        -e "s/^ALLOCATION_STRATEGY = .*$/ALLOCATION_STRATEGY = '${cfg_prop_list[19]}'/" \
        -e "s/^CUSTOM_ALLOCATION = .*$/CUSTOM_ALLOCATION = ${cfg_prop_list[20]}/" \
        auto_script/config_template.py

    # remove quotes
    sed -i -e "s/'np.inf'/np.inf/g" auto_script/config_template.py
    
    cp auto_script/config_template.py ../core/config.py

    python3 run_experiment.py -t "${cfg_prop_list[1]}" -o $root_dir/"${cfg_prop_list[2]}" &> log.txt

    mv log.txt $root_dir/"${cfg_prop_list[2]}"
    cp ../core/config.py $root_dir/"${cfg_prop_list[2]}"

    mv auto_script/config_template.py.bak auto_script/config_template.py
}

mkdir -p results

if [ ! -d "exps/1" ]; then
    mkdir -p exps/1

    cd exps/1
    git clone https://github.com/tamitakada/VortexScheduler
    cd VortexScheduler
    cd $root_dir

    cp -r exps/1 exps/2
    cp -r exps/1 exps/3
    cp -r exps/1 exps/4
    cp -r exps/1 exps/5
fi

prop_list=()
i=0
IFS=$'\n'
while read -r property; do
    if [[ -z "$property" ]]; then
        clone_num=$((i % 5 + 1))
        echo "Current clone: $clone_num"

        if [[ "$clone_num" -eq "1" && "$i" -ne "0" ]]; then
            echo "Waiting for previous to finish..."
            wait
        fi

        cd $root_dir/exps/$clone_num/VortexScheduler
        cd cluster_simulation
        source set_env.sh
        export PYTHONPATH="${SIMULATION_DIR}"
        cd experiments

        echo "Running config..."
        run_experiment "${prop_list[@]}" &

        cd $root_dir

        prop_list=()
        ((i++))
    else
        prop_list+=("$property")
    fi
done <<< "$configs_out"

if [ "${#prop_list[@]}" -gt 0 ]; then
    clone_num=$((i % 5 + 1))
    echo "Current clone: $clone_num"

    if [[ "$clone_num" -eq "1" && "$i" -ne "0" ]]; then
        echo "Waiting for previous to finish..."
        wait
    fi

    cd $root_dir/exps/$clone_num/VortexScheduler
    cd cluster_simulation
    source set_env.sh
    export PYTHONPATH="${SIMULATION_DIR}"
    cd experiments

    echo "Running config..."
    run_experiment "${prop_list[@]}" &
fi

echo "Started all configs. Waiting for tasks to finish..."
wait