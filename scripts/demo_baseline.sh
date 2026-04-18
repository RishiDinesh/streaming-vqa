#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_FROM_SCRIPT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ROOT=${ROOT:-${ROOT_FROM_SCRIPT}}
PARENT_DIR=$(cd -- "${ROOT}/.." && pwd)

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TERM=${TERM:-xterm-256color}
export FORCE_COLOR=${FORCE_COLOR:-1}
export CLICOLOR_FORCE=${CLICOLOR_FORCE:-1}
export PY_COLORS=${PY_COLORS:-1}

DEFAULT_PYTHON_BIN="${PARENT_DIR}/.conda/envs/duo/bin/python"
PYTHON_BIN=${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Could not find the duo environment python at ${PYTHON_BIN}." >&2
    echo "Set PYTHON_BIN explicitly to the python inside the 'duo' Conda environment." >&2
    exit 1
fi


TIME_LIMIT=${TIME_LIMIT:-01:00:00}
JOB_NAME=${JOB_NAME:-live-llava-baseline}
MODEL_NAME=${MODEL_NAME:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
VIDEO_PATH=${VIDEO_PATH:-data/sample.mp4}
NUM_FRAMES=${NUM_FRAMES:-32}
MAX_LENGTH=${MAX_LENGTH:-22000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
PROMPT=${PROMPT:-Describe this video in detail.}
ATTENTION_MODE=${ATTENTION_MODE:-baseline}
REPORT_JSON=${REPORT_JSON:-}

cd "${ROOT}"

CMD=(
    "${PYTHON_BIN}"
    "${ROOT}/live_llava_video.py"
    --model_name "${MODEL_NAME}"
    --video_path "${VIDEO_PATH}"
    --num_frames "${NUM_FRAMES}"
    --max_length "${MAX_LENGTH}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --prompt "${PROMPT}"
    --attention_mode "${ATTENTION_MODE}"
)

if [[ -n "${REPORT_JSON}" ]]; then
    CMD+=(--report_json "${REPORT_JSON}")
fi

exec srun \
    --partition=gpunodes \
    --gres=gpu:rtx_a4500:1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=20G \
    --time=1-00:00:00 \
    --job-name="${JOB_NAME}" \
    "${CMD[@]}" \
    "$@"
