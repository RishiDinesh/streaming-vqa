#!/usr/bin/env bash
# SLURM array smoke test: 7B model, 1 video, all 4 methods, rvs-movie only.
# Array index 0-3 maps to one method each (1 GPU per job).
# Purpose: verify no OOM before submitting full 80-job eval.
#
# Usage (submit from repo root):
#   sbatch --array=0-3 --output="logs/7b-smoke-%a-%j.out" scripts/run_7b_smoke_test.sh
#
#SBATCH --job-name=7b-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
mkdir -p "${ROOT}/logs"

export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"

source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"

MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
DATASET="rvs_movie"
ATTN_DIR="outputs/train/7b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5"
SPARSITY="0.75"
DEPLOY_SINK_SIZE="256"
DEPLOY_RECENT_SIZE="512"
OUTPUT_ROOT="${ROOT}/outputs/evaluations_streaming/7b_smoke"

METHODS=(full_streaming duo_streaming rekv duo_plus_rekv)
METHOD="${METHODS[${SLURM_ARRAY_TASK_ID}]}"

OUTPUT_PATH="${OUTPUT_ROOT}/${DATASET}/${METHOD}/chunk_000.json"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "[7B smoke ${SLURM_ARRAY_TASK_ID}] method=${METHOD} model=${MODEL} dataset=${DATASET}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --hf-repo-id "Becomebright/RVS"
  --allow-hf-video-download
  --model "${MODEL}"
  --method "${METHOD}"
  --sample-fps 0.5
  --max-new-tokens 64
  --video-decode-threads 1
  --max-videos 1
  --output-path "${OUTPUT_PATH}"
)

case "${METHOD}" in
  duo_streaming)
    COMMON_ARGS+=(--attn-dir "${ATTN_DIR}" --sparsity "${SPARSITY}"
                  --deploy-sink-size "${DEPLOY_SINK_SIZE}"
                  --deploy-recent-size "${DEPLOY_RECENT_SIZE}")
    ;;
  rekv)
    COMMON_ARGS+=(--retrieve-size 64 --n-local 15000)
    ;;
  duo_plus_rekv)
    COMMON_ARGS+=(--attn-dir "${ATTN_DIR}" --sparsity "${SPARSITY}"
                  --retrieve-size 64 --n-local 15000
                  --deploy-sink-size "${DEPLOY_SINK_SIZE}"
                  --deploy-recent-size "${DEPLOY_RECENT_SIZE}")
    ;;
esac

python -m streaming.ReKV.run_eval "${COMMON_ARGS[@]}"

echo "[7B smoke ${SLURM_ARRAY_TASK_ID}] done: ${METHOD}"
