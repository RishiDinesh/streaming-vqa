#!/bin/bash
#SBATCH --job-name=eval-llava
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodelist=gpunode2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

cd /w/nobackup/385/scratch-space/expires-2026-Mar-27/mihir/streaming-vqa

# MMDuo visual 
# /w/nobackup/385/scratch-space/expires-2026-Mar-27/mihir/miniconda3_fresh/envs/duo/bin/python -m duo_attn.eval.efficiency.benchmark_dynamic_llava \
#   --model_name models/ \
#   --video_path data/sample.mp4 \
#   --num_frames 8 \
#   --attn_load_dir weights/ \
#   --sparsity 0.5 \
#   --output_dir ./benchmark_results_llava \
#   --ui_style demo \
#   --prefill_chunk_size 32000 \
#   --decode_tokens 100\
#   --prompt "$(cat long_prompt.txt)" \
#   --num_frames 32 \

# MMDuo
# /w/nobackup/385/scratch-space/expires-2026-Mar-27/mihir/miniconda3_fresh/envs/duo/bin/python -m duo_attn.eval.efficiency.benchmark_dynamic_llava \
#   --model_name models/ \
#   --dataset_type egoschema \
#   --video_root data/videos \
#   --annotation_path data/questions.json \
#   --batch_size 5 \
#   --num_frames 64 \
#   --ui_style benchmark \
#   --output_dir benchmark_results_llava \
#   --attention_mode duo \
#   --attn_load_dir weights \
#   --sparsity 0.5

# Baseline
/w/nobackup/385/scratch-space/expires-2026-Mar-27/mihir/miniconda3_fresh/envs/duo/bin/python -m duo_attn.eval.efficiency.benchmark_dynamic_llava \
  --model_name models/ \
  --dataset_type egoschema \
  --video_root data/videos \
  --annotation_path data/questions.json \
  --batch_size 5 \
  --num_frames 64 \
  --ui_style benchmark \
  --output_dir benchmark_results_llava \
  --attention_mode baseline\
  --sparsity 0.5
