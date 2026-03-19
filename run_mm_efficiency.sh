#!/bin/bash

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Mar-27/rishi/streaming-vqa
cd "${ROOT}"
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-/w/nobackup/385/scratch-space/expires-2026-Mar-27/rishi/.conda/envs/mmda-cuda124/bin/python}
MODEL_NAME=${MODEL_NAME:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
ATTN_LOAD_DIR=${ATTN_LOAD_DIR:-./untracked/llava_ov_final_blocksparse/}
VIDEO_ROOT=${VIDEO_ROOT:-./vnbench_data/VNBench_new}
ANNOTATION_PATH=${ANNOTATION_PATH:-./vnbench_data/anno.jsonl}
MAX_LENGTH=${MAX_LENGTH:-32768}
NUM_FRAMES=${NUM_FRAMES:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
SPARSITY=${SPARSITY:-0.5}
PREFILLING_CHUNK_SIZE=${PREFILLING_CHUNK_SIZE:-32000}

OUT_DYNAMIC_BASELINE=${OUT_DYNAMIC_BASELINE:-./untracked/mm_efficiency_compare/dynamic_baseline}
OUT_DYNAMIC_DUO=${OUT_DYNAMIC_DUO:-./untracked/mm_efficiency_compare/dynamic_duo}
OUT_STATIC_FULL_PROXY=${OUT_STATIC_FULL_PROXY:-./untracked/mm_efficiency_compare/static_full_proxy}
OUT_STATIC_DUO=${OUT_STATIC_DUO:-./untracked/mm_efficiency_compare/static_duo}

submit_gpu_job() {
  local job_name="$1"
  local wrap_cmd="$2"
  sbatch --parsable \
    --job-name="${job_name}" \
    --partition=gpunodes \
    --gres=gpu:rtx_a6000:1 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=10G \
    --time=1-00:00:00 \
    --output=logs/%x-%j.out \
    --wrap "${wrap_cmd}"
}

dynamic_baseline_cmd=$(cat <<EOF
set -eu
cd "${ROOT}"
mkdir -p "${OUT_DYNAMIC_BASELINE}"
"${PYTHON_BIN}" -u duo_attn/eval/efficiency/benchmark_dynamic.py \
  --model_name "${MODEL_NAME}" \
  --video_root "${VIDEO_ROOT}" \
  --annotation_path "${ANNOTATION_PATH}" \
  --max_length "${MAX_LENGTH}" \
  --num_frames "${NUM_FRAMES}" \
  --num_workers "${NUM_WORKERS}" \
  --output_dir "${OUT_DYNAMIC_BASELINE}"
EOF
)

dynamic_duo_cmd=$(cat <<EOF
set -eu
cd "${ROOT}"
mkdir -p "${OUT_DYNAMIC_DUO}"
"${PYTHON_BIN}" -u duo_attn/eval/efficiency/benchmark_dynamic.py \
  --model_name "${MODEL_NAME}" \
  --attn_load_dir "${ATTN_LOAD_DIR}" \
  --video_root "${VIDEO_ROOT}" \
  --annotation_path "${ANNOTATION_PATH}" \
  --max_length "${MAX_LENGTH}" \
  --num_frames "${NUM_FRAMES}" \
  --num_workers "${NUM_WORKERS}" \
  --sparsity "${SPARSITY}" \
  --output_dir "${OUT_DYNAMIC_DUO}"
EOF
)

static_full_proxy_cmd=$(cat <<EOF
set -eu
cd "${ROOT}"
mkdir -p "${OUT_STATIC_FULL_PROXY}"
"${PYTHON_BIN}" -u duo_attn/eval/efficiency/benchmark_static.py \
  --model_name "${MODEL_NAME}" \
  --attn_load_dir "${ATTN_LOAD_DIR}" \
  --video_root "${VIDEO_ROOT}" \
  --annotation_path "${ANNOTATION_PATH}" \
  --max_length "${MAX_LENGTH}" \
  --num_frames "${NUM_FRAMES}" \
  --num_workers "${NUM_WORKERS}" \
  --sparsity 0 \
  --prefilling_chunk_size "${PREFILLING_CHUNK_SIZE}" \
  --output_dir "${OUT_STATIC_FULL_PROXY}"
EOF
)

static_duo_cmd=$(cat <<EOF
set -eu
cd "${ROOT}"
mkdir -p "${OUT_STATIC_DUO}"
"${PYTHON_BIN}" -u duo_attn/eval/efficiency/benchmark_static.py \
  --model_name "${MODEL_NAME}" \
  --attn_load_dir "${ATTN_LOAD_DIR}" \
  --video_root "${VIDEO_ROOT}" \
  --annotation_path "${ANNOTATION_PATH}" \
  --max_length "${MAX_LENGTH}" \
  --num_frames "${NUM_FRAMES}" \
  --num_workers "${NUM_WORKERS}" \
  --sparsity "${SPARSITY}" \
  --prefilling_chunk_size "${PREFILLING_CHUNK_SIZE}" \
  --output_dir "${OUT_STATIC_DUO}"
EOF
)

compare_cmd=$(cat <<EOF
set -eu
cd "${ROOT}"
"${PYTHON_BIN}" -u compare_mm_efficiency.py
EOF
)

job1=$(submit_gpu_job eff_dynamic_baseline "${dynamic_baseline_cmd}")
echo "Submitted dynamic baseline as job ${job1}"

job2=$(submit_gpu_job eff_dynamic_duo "${dynamic_duo_cmd}")
echo "Submitted dynamic duo as job ${job2}"

job3=$(submit_gpu_job eff_static_full_proxy "${static_full_proxy_cmd}")
echo "Submitted static full proxy as job ${job3}"

job4=$(submit_gpu_job eff_static_duo "${static_duo_cmd}")
echo "Submitted static duo as job ${job4}"

job5=$(
  sbatch --parsable \
    --dependency=afterok:${job1}:${job2}:${job3}:${job4} \
    --job-name=eff_compare \
    --partition=gpunodes \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=4G \
    --time=00:15:00 \
    --output=logs/%x-%j.out \
    --wrap "${compare_cmd}"
)

echo "Submitted compare_efficiency_table.py as job ${job5} (afterok:${job1}:${job2}:${job3}:${job4})"
echo "All jobs submitted."
