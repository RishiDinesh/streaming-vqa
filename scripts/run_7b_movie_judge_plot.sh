#!/usr/bin/env bash
# Merge rvs-movie 7B chunks, run judge scoring (array 0-3), then plot.
# Step 1 (merge) runs on login node inline before submitting judge jobs.
# Step 2 (judge) is a SLURM array 0-3, one method per task.
# Step 3 (plot) runs after all judge tasks complete (dependency afterok).
#
# Usage (from repo root):
#   bash scripts/run_7b_movie_judge_plot.sh
#
#SBATCH --job-name=7b-movie-judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"

source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"

JUDGE_MODEL="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
MERGED_DIR="${ROOT}/outputs/evaluations_streaming/rvs-movie/7b_full_eval/merged"
METHODS=(full_streaming duo_streaming rekv duo_plus_rekv)

IDX=${SLURM_ARRAY_TASK_ID}
METHOD="${METHODS[$IDX]}"
RESULT_FILE="${MERGED_DIR}/${METHOD}.json"

echo "[7b-movie-judge ${IDX}] scoring: ${RESULT_FILE}"
python -m streaming.ReKV.judge_results \
  --judge-model "${JUDGE_MODEL}" \
  --dtype bfloat16 \
  --in-place \
  "${RESULT_FILE}"
echo "[7b-movie-judge ${IDX}] done: ${METHOD}"
