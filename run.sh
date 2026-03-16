#!/bin/bash
#SBATCH --job-name=test-llava-ov-1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

cd /w/nobackup/385/scratch-space/expires-2026-Mar-27/rishi/streaming-vqa
/w/nobackup/385/scratch-space/expires-2026-Mar-27/rishi/.conda/envs/mmda-cuda124/bin/python run_sample.py --num_frames 64 --num_samples 5