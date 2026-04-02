#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_FROM_SCRIPT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_FROM_SCRIPT}"

PRETRAINED="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
MODE="duo"
SPARSITY="0.5"
MAX_FRAMES_NUM="64"
DECODING_SIMULATION_LENGTH="256"
RUN_SCRIPT="eval/run_dataset.sh"
SBATCH_ARGS=(--gres=gpu:rtx_4090:1 --mem=30G)

SINK_VALUES=(64 128 256 512)
RECENT_VALUES=(128 256 512 1024)

append_optional_env_var() {
    local name=$1
    if [[ -n "${!name:-}" ]]; then
        ENV_ARGS+=("${name}=${!name}")
    fi
}

ENV_ARGS=(
    "PRETRAINED=${PRETRAINED}"
    "ATTN_DIR=${ATTN_DIR}"
    "MODE=${MODE}"
    "SPARSITY=${SPARSITY}"
    "MAX_FRAMES_NUM=${MAX_FRAMES_NUM}"
    "DECODING_SIMULATION_LENGTH=${DECODING_SIMULATION_LENGTH}"
)

append_optional_env_var LIMIT
append_optional_env_var PYTHON_BIN
append_optional_env_var ROOT
append_optional_env_var DEVICE_MAP
append_optional_env_var NUM_PROCESSES
append_optional_env_var NUM_MACHINES
append_optional_env_var MAIN_PROCESS_PORT
append_optional_env_var SAME_NETWORK
append_optional_env_var CAP_FPS_SAMPLING
append_optional_env_var LOG_SAMPLES

submitted=0
for sink in "${SINK_VALUES[@]}"; do
    for recent in "${RECENT_VALUES[@]}"; do
        if (( recent < sink )); then
            continue
        fi

        config_name="fr${MAX_FRAMES_NUM}-r${recent}-s${sink}-de${DECODING_SIMULATION_LENGTH}"
        output_name="egoschema-sweep/${config_name}-sp50"
        job_name="egosweep-${config_name}-sp50"

        cmd=(
            env
            "${ENV_ARGS[@]}"
            "DEPLOY_SINK_SIZE=${sink}"
            "DEPLOY_RECENT_SIZE=${recent}"
            "OUTPUT_NAME=${output_name}"
            sbatch
            "${SBATCH_ARGS[@]}"
            --job-name "${job_name}"
            "${RUN_SCRIPT}"
            egoschema_subset
        )

        echo "Submitting ${job_name} -> ${output_name}"
        "${cmd[@]}"
        submitted=$((submitted + 1))
    done
done

echo "Submitted ${submitted} EgoSchema sink/recent sweep jobs."
