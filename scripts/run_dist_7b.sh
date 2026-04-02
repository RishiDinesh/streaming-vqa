#!/bin/bash
#SBATCH --job-name=train-7b
#SBATCH --partition=gpunodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=rishidinesh@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

cd /w/nobackup/385/scratch-space/expires-2026-Mar-27/rishi/streaming-vqa

TORCHRUN_BIN=/w/nobackup/385/scratch-space/expires-2026-Mar-27/rishi/.conda/envs/mmda-cuda124/bin/torchrun
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

srun bash -lc '
  '"$TORCHRUN_BIN"' \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    --rdzv_id=$SLURM_JOB_ID \
    duo_attn/train.py \
      --dataset_format video_qa \
      --dataset_name dynamic_synthetic \
      --max_length 32000 \
      --batch_size 1 \
      --video_root ./datasets/unedited_500 \
      --num_frames 64 \
      --min_needle_depth_ratio 0.1 \
      --max_needle_depth_ratio 0.8 \
      --num_needles 1 \
      --model_name llava-hf/llava-onevision-qwen2-7b-ov-hf \
      --num_steps 500 \
      --sink_size 512 \
      --recent_size 1024 \
      --streaming_attn_implementation blocksparse
'
