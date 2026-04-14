#!/usr/bin/env bash
# Run full_streaming for rvs-movie chunk_004, skipping tt0121765 (OOM video).
# Runs tt0167190 and tt0186151 individually via --video-id.
# Array index: 0 -> tt0167190, 1 -> tt0186151
#
#SBATCH --job-name=movie-fs-c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"

source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"

MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
VIDEO_IDS=(tt0167190 tt0186151)
VIDEO_ID="${VIDEO_IDS[${SLURM_ARRAY_TASK_ID}]}"

OUTPUT_PATH="${ROOT}/outputs/evaluations_streaming/rvs-movie/7b_full_eval/full_streaming/chunk_004_${VIDEO_ID}.json"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "[movie-fs-c4 ${SLURM_ARRAY_TASK_ID}] video=${VIDEO_ID}"

python -m streaming.ReKV.run_eval \
  --dataset rvs_movie \
  --hf-repo-id "Becomebright/RVS" \
  --allow-hf-video-download \
  --model "${MODEL}" \
  --method full_streaming \
  --sample-fps 0.5 \
  --max-new-tokens 64 \
  --video-decode-threads 1 \
  --video-id "${VIDEO_ID}" \
  --output-path "${OUTPUT_PATH}"

echo "[movie-fs-c4 ${SLURM_ARRAY_TASK_ID}] done: ${VIDEO_ID}"
