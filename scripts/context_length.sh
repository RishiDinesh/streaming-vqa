#!/bin/bash
#SBATCH --job-name=ctx-length
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

python -u duo_attn/eval/efficiency/context_eval_llava.py run \
  --model_scale 0.5b \
  --output_dir ./outputs/benchmarking/llava-ov-0.5b \
  --attn_load_dir ./outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632 \
  # --attn_load_dir ./outputs/train/7b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5 \
  --prompt "Describe this video in detail."   \
  --max_length 32000   \
  --max_context 32000   \
  --target_contexts 4000 8000 12000 16000 20000 24000 28000 32000   \
  --decode_tokens 100   \
  --sparsity 0.5


# python -m duo_attn.eval.efficiency.context_eval_llava plot \
#   --input_json untracked/context_sweep_32k/context_sweep_results.json
