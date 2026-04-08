#!/bin/bash
#SBATCH --job-name=eval-llava
#SBATCH --partition=gpunodes
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-/root/miniconda3/envs/duo/bin/python}
PROMPT_FILE=${PROMPT_FILE:-"${ROOT_DIR}/long_prompt.txt"}
MODEL_DIR=${MODEL_DIR:-"${ROOT_DIR}/models/llava-hf/llava-onevision-qwen2-0.5b-ov-hf"}
ATTN_DIR=${ATTN_DIR:-"${ROOT_DIR}/outputs/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1"}
VIDEO_PATH=${VIDEO_PATH:-"${ROOT_DIR}/data/sample.mp4"}
PROMPT_TEXT=${PROMPT_TEXT:-"Describe this video in detail."}

if [ -f "${PROMPT_FILE}" ]; then
  PROMPT_TEXT="$(cat "${PROMPT_FILE}")"
fi

cd "${ROOT_DIR}"

# MMDuo visual 
$PYTHON -m duo_attn.eval.efficiency.benchmark_dynamic_llava \
  --model_name "${MODEL_DIR}" \
  --video_path "${VIDEO_PATH}" \
  --num_frames 8 \
  --attn_load_dir "${ATTN_DIR}" \
  --sparsity 0.5 \
  --output_dir "${ROOT_DIR}/benchmark_results_llava" \
  --ui_style demo \
  --prefill_chunk_size 32000 \
  --decode_tokens 100 \
  --prompt "${PROMPT_TEXT}" \
  --num_frames 32\
  --attention_mode baseline

# MMDuo
# $PYTHON -m duo_attn.eval.efficiency.benchmark_dynamic_llava \
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
# $PYTHON -m duo_attn.eval.efficiency.benchmark_dynamic_llava \
#   --model_name models/ \
#   --dataset_type egoschema \
#   --video_root data/videos \
#   --annotation_path data/questions.json \
#   --batch_size 5 \
#   --num_frames 64 \
#   --ui_style benchmark \
#   --output_dir benchmark_results_llava \
#   --attention_mode baseline
