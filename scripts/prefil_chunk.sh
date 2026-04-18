#!/bin/bash
#SBATCH --job-name=prefill-chunk
#SBATCH --partition=gpunodes
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

python -u -m duo_attn.eval.efficiency.prefill_eval_llava prefill \
  --model-scale 7b \
  --output_dir ./outputs/benchmarking/llava-ov-7b/prefill_chunk_sweep_32k \
  --attn_load_dir ./outputs/train/7b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5 \
  --prompt_file long_prompt.txt \
  --target_context 32000 \
  --max_length 32000 \
  --max_num_frames 512 \
  --prefill_chunk_sizes 4000 8000 12000 16000 20000 24000 28000 32000 \
  --threshold 0.5 \
  --sparsity 0.5

# python -u -m duo_attn.eval.efficiency.prefill_eval_llava context \
#   --model-scale 0.5b \
#   --output_dir untracked/context_sweep_32k \
#   --attn_load_dir outputs/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1 \
#   --prompt_file long_prompt.txt \
#   --max_length 32000 \
#   --max_context 32000 \
#   --target_contexts 4000 8000 12000 16000 20000 24000 28000 32000
