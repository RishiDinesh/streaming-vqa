#!/bin/bash
#SBATCH --job-name=0p5b-val
#SBATCH --partition=gpunodes
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=rishidinesh@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

BATCH_SIZE=${BATCH_SIZE:-1}
N_SAMPLES=${N_SAMPLES:-50}
MAX_LENGTH=${MAX_LENGTH:-32000}
NUM_NODES=${NUM_NODES:-4}
ATTN_LOAD_DIR=${ATTN_LOAD_DIR:-./outputs/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1}
DTYPE=${DTYPE:-float16}

EXP_NAME=$(basename "${ATTN_LOAD_DIR%/}")
VAL_DIR=./outputs/validation/${EXP_NAME}_val
mkdir -p "$VAL_DIR"

TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29511}

srun bash -lc '
  '"$TORCHRUN_BIN"' \
    --nnodes='"$NUM_NODES"' \
    --nproc_per_node=1 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    --rdzv_id=$SLURM_JOB_ID \
    duo_attn/eval/validate/runner.py \
      --model_name llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
      --attn_load_dir '"$ATTN_LOAD_DIR"' \
      --video_root ./datasets/vnbench/videos \
      --annotation_path ./datasets/vnbench/anno.jsonl \
      --batch_size '"$BATCH_SIZE"' \
      --dtype '"$DTYPE"' \
      --max_length '"$MAX_LENGTH"' \
      --n_samples '"$N_SAMPLES"' \
      --sparsity 0.5 \
      --output_json '"$VAL_DIR"'/retrieval_pool_ratio_ablation.json \
      --output_csv '"$VAL_DIR"'/retrieval_pool_ratio_ablation.csv \
      --output_plot '"$VAL_DIR"'/retrieval_pool_ratio_ablation_plot.png
'
