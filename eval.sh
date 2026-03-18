#!/bin/bash
#SBATCH --job-name=eval-llava
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

cd /w/nobackup/385/scratch-space/expires-2026-Mar-27/mihir/streaming-vqa

/w/nobackup/385/scratch-space/expires-2026-Mar-27/mihir/miniconda3_fresh/envs/duo/bin/python -m duo_attn.eval.efficiency.benchmark_dynamic_llava \
        --model_name models/ \
        --video_path data/sample.mp4 \
        --num_frames 8 \
        --attn_load_dir weights/ \
        --sparsity 0.5 \
        --output_dir ./benchmark_results_llava


# squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"    # check queue