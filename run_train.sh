#!/bin/bash
#SBATCH --job-name=train-llava-ov-pilot
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd /w/nobackup/385/scratch-space/expires-2026-Mar-27/rishi/streaming-vqa

MODEL_NAME=${MODEL_NAME:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
VIDEO_ROOT=${VIDEO_ROOT:-./vnbench_data/VNBench_new/}
ANNOTATION_PATH=${ANNOTATION_PATH:-./vnbench_data/anno.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./untracked/llava_ov_final_blocksparse}
NUM_STEPS=${NUM_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-50}
LR=${LR:-1e-1}

/w/nobackup/385/scratch-space/expires-2026-Mar-27/rishi/.conda/envs/mmda-cuda124/bin/torchrun --nproc_per_node=1 duo_attn/train.py \
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
  --streaming_attn_implementation blocksparse \
  --output_dir "${OUTPUT_DIR}" \
  --save_steps "${SAVE_STEPS}"