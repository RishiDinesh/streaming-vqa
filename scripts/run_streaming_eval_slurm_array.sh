#!/usr/bin/env bash
# Multi-GPU sharded SLURM array job for streaming ReKV evaluation.
# Each array task handles a contiguous chunk of videos on one GPU.
#
# Usage (submit from repo root):
#   NUM_CHUNKS=8 DATASET=rvs_ego METHOD=rekv \
#     sbatch --array=0-7 scripts/run_streaming_eval_slurm_array.sh
#
# Key environment variables (all have sensible defaults):
#   DATASET            rvs_ego | rvs_movie           (default: rvs_movie)
#   METHOD             full_streaming | duo_streaming | rekv | duo_plus_rekv
#                                                     (default: rekv)
#   NUM_CHUNKS         number of shards               (default: 1)
#   MODEL              HF model path                  (default: llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
#   SAMPLE_FPS         frames per second to sample    (default: 0.5)
#   MAX_NEW_TOKENS     max tokens to generate         (default: 64)
#   VIDEO_DECODE_THREADS  decord threads per worker   (default: 4)
#   OUTPUT_ROOT        base output directory          (default: outputs/evaluations_streaming/...)
#   FEATURE_CACHE_ROOT path to feature cache          (only used when USE_FEATURE_CACHE=1)
#   ATTN_DIR           path to Duo attention weights  (required for duo_* methods)
#   SPARSITY           Duo head sparsity              (default: 0.5)
#   DEPLOY_SINK_SIZE   Duo sink window override       (default: use trained value)
#   DEPLOY_RECENT_SIZE Duo recent window override     (default: use trained value)
#   REKV_TOPK          ReKV retrieved blocks          (default: 64)
#   REKV_N_LOCAL       ReKV local window size tokens  (default: 15000)
#   DUO_STRICT_NO_SDPA_FALLBACK  fail if block_sparse_attn missing (default: 1)
#   RESUME             resume previous run            (default: 1)
#   EXTRA_ARGS         extra args forwarded to run_eval.py
#
#SBATCH --job-name=stream-eval-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

# Hardcode ROOT: BASH_SOURCE[0] is unreliable when SLURM copies the script to spool.
ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"

# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required. Submit this script with sbatch --array=..." >&2
  exit 1
fi

DATASET=${DATASET:-rvs_movie}
MODEL=${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
METHOD=${METHOD:-rekv}
HF_REPO_ID=${HF_REPO_ID:-Becomebright/RVS}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-1}
NUM_CHUNKS=${NUM_CHUNKS:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/outputs/evaluations_streaming/${DATASET//_/-}/slurm_array}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-}
USE_FEATURE_CACHE=${USE_FEATURE_CACHE:-0}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
SPARSITY=${SPARSITY:-0.5}
DEPLOY_SINK_SIZE=${DEPLOY_SINK_SIZE:-}
DEPLOY_RECENT_SIZE=${DEPLOY_RECENT_SIZE:-}
REKV_TOPK=${REKV_TOPK:-64}
REKV_N_LOCAL=${REKV_N_LOCAL:-15000}
DUO_STRICT_NO_SDPA_FALLBACK=${DUO_STRICT_NO_SDPA_FALLBACK:-1}
RESUME=${RESUME:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
if (( CHUNK_INDEX >= NUM_CHUNKS )); then
  echo "SLURM_ARRAY_TASK_ID=${CHUNK_INDEX} must be smaller than NUM_CHUNKS=${NUM_CHUNKS}" >&2
  exit 1
fi

OUTPUT_PATH="${OUTPUT_ROOT}/${METHOD}/chunk_$(printf '%03d' "${CHUNK_INDEX}").json"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "[array task ${CHUNK_INDEX}/${NUM_CHUNKS}] dataset=${DATASET} method=${METHOD} output=${OUTPUT_PATH}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --hf-repo-id "${HF_REPO_ID}"
  --allow-hf-video-download
  --model "${MODEL}"
  --method "${METHOD}"
  --sample-fps "${SAMPLE_FPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --video-decode-threads "${VIDEO_DECODE_THREADS}"
  --num-chunks "${NUM_CHUNKS}"
  --chunk-index "${CHUNK_INDEX}"
  --output-path "${OUTPUT_PATH}"
)

if [[ "${USE_FEATURE_CACHE}" == "1" && -n "${FEATURE_CACHE_ROOT}" ]]; then
  COMMON_ARGS+=(--feature-cache-root "${FEATURE_CACHE_ROOT}")
fi

if [[ "${RESUME}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

if [[ "${DUO_STRICT_NO_SDPA_FALLBACK}" == "1" ]]; then
  COMMON_ARGS+=(--duo-strict-no-sdpa-fallback)
fi

case "${METHOD}" in
  duo_streaming)
    COMMON_ARGS+=(--attn-dir "${ATTN_DIR}" --sparsity "${SPARSITY}")
    [[ -n "${DEPLOY_SINK_SIZE}"   ]] && COMMON_ARGS+=(--deploy-sink-size   "${DEPLOY_SINK_SIZE}")
    [[ -n "${DEPLOY_RECENT_SIZE}" ]] && COMMON_ARGS+=(--deploy-recent-size "${DEPLOY_RECENT_SIZE}")
    ;;
  rekv)
    COMMON_ARGS+=(--retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}")
    ;;
  duo_plus_rekv)
    COMMON_ARGS+=(--attn-dir "${ATTN_DIR}" --sparsity "${SPARSITY}" --retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}")
    [[ -n "${DEPLOY_SINK_SIZE}"   ]] && COMMON_ARGS+=(--deploy-sink-size   "${DEPLOY_SINK_SIZE}")
    [[ -n "${DEPLOY_RECENT_SIZE}" ]] && COMMON_ARGS+=(--deploy-recent-size "${DEPLOY_RECENT_SIZE}")
    ;;
esac

python -m streaming.ReKV.run_eval \
  "${COMMON_ARGS[@]}" \
  ${EXTRA_ARGS}
