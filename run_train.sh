#!/bin/bash
#SBATCH --job-name=train-llava-ov-pilot
#SBATCH --partition=gpunodes
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"

MODEL_NAME=${MODEL_NAME:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
VIDEO_ROOT=${VIDEO_ROOT:-./vnbench_data/VNBench_new/}
ANNOTATION_PATH=${ANNOTATION_PATH:-./vnbench_data/anno.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./untracked/llava_ov_final_blocksparse}
NUM_STEPS=${NUM_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-50}
LR=${LR:-1e-1}

TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}

"${TORCHRUN_BIN}" --nproc_per_node=1 --module duo_attn.train \
  --model_name "${MODEL_NAME}" \
  --dataset_format video_qa \
  --video_root "${VIDEO_ROOT}" \
  --annotation_path "${ANNOTATION_PATH}" \
  --num_steps "${NUM_STEPS}" \
  --batch_size 1 \
  --max_length 32768 \
  --num_frames 64 \
  --num_workers 4 \
  --lr "${LR}" \
  --streaming_attn_implementation auto \
  --output_dir "${OUTPUT_DIR}" \
  --save_steps "${SAVE_STEPS}"
