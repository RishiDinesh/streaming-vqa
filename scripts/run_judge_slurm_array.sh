#!/usr/bin/env bash
# SLURM array job to run LLM judge scoring on all merged result JSONs.
# Array index maps to a specific (dataset, method) combination.
#
# Usage (submit from repo root):
#   sbatch --array=0-7 scripts/run_judge_slurm_array.sh
#
#SBATCH --job-name=judge-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"

source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"

JUDGE_MODEL=${JUDGE_MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}

# Array index → result JSON path (0-3: movie, 4-7: ego)
RESULT_FILES=(
  "outputs/evaluations_streaming/rvs-movie/full_eval/merged_sink256/full_streaming.json"
  "outputs/evaluations_streaming/rvs-movie/full_eval/merged_sink256/duo_streaming.json"
  "outputs/evaluations_streaming/rvs-movie/full_eval/merged_sink256/rekv.json"
  "outputs/evaluations_streaming/rvs-movie/full_eval/merged_sink256/duo_plus_rekv.json"
  "outputs/evaluations_streaming/rvs-ego/full_eval/run2/merged/full_streaming.json"
  "outputs/evaluations_streaming/rvs-ego/full_eval/run2/merged/duo_streaming.json"
  "outputs/evaluations_streaming/rvs-ego/full_eval/run2/merged/rekv.json"
  "outputs/evaluations_streaming/rvs-ego/full_eval/run2/merged/duo_plus_rekv.json"
)

IDX=${SLURM_ARRAY_TASK_ID}
RESULT_FILE="${ROOT}/${RESULT_FILES[$IDX]}"

echo "[judge task ${IDX}] scoring: ${RESULT_FILE}"
echo "[judge task ${IDX}] judge model: ${JUDGE_MODEL}"

python -m streaming.ReKV.judge_results \
  --judge-model "${JUDGE_MODEL}" \
  --dtype bfloat16 \
  --in-place \
  "${RESULT_FILE}"

echo "[judge task ${IDX}] done."
