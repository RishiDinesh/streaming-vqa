#!/bin/bash
#SBATCH --job-name=train-dist
#SBATCH --partition=gpunodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx_a4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd /w/nobackup/385/scratch-space/expires-2026-Mar-27/rishi/streaming-vqa
mkdir -p logs ./untracked/llava_ov_dist_smoke

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
      --model_name llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
      --dataset_format video_qa \
      --video_root ./vnbench_data/VNBench_new \
      --annotation_path ./vnbench_data/anno.jsonl \
      --num_steps 5 \
      --batch_size 1 \
      --max_length 15000 \
      --num_frames 8 \
      --num_workers 4 \
      --lr 1e-2 \
      --streaming_attn_implementation blocksparse \
      --output_dir ./untracked/llava_ov_dist_smoke \
      --save_steps 5 \
      --disable_wandb
'