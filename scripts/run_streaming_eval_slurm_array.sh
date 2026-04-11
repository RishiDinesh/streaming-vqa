#!/usr/bin/env bash
#SBATCH --job-name=stream-eval-array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${ROOT}/logs"

# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"
activate_streaming_env

cd "${ROOT}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

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
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-4}
NUM_CHUNKS=${NUM_CHUNKS:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/evaluations_streaming/${DATASET//_/-}/slurm_array}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-}
USE_FEATURE_CACHE=${USE_FEATURE_CACHE:-0}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
SPARSITY=${SPARSITY:-0.5}
REKV_TOPK=${REKV_TOPK:-64}
REKV_N_LOCAL=${REKV_N_LOCAL:-15000}
DUO_STRICT_NO_SDPA_FALLBACK=${DUO_STRICT_NO_SDPA_FALLBACK:-0}
RESUME=${RESUME:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

CHUNK_INDEX=${SLURM_ARRAY_TASK_ID}
if (( CHUNK_INDEX >= NUM_CHUNKS )); then
  echo "SLURM_ARRAY_TASK_ID=${CHUNK_INDEX} must be smaller than NUM_CHUNKS=${NUM_CHUNKS}" >&2
  exit 1
fi

OUTPUT_PATH="${OUTPUT_ROOT}/${METHOD}/chunk_${CHUNK_INDEX}.json"

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
    ;;
  rekv)
    COMMON_ARGS+=(--retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}")
    ;;
  duo_plus_rekv)
    COMMON_ARGS+=(--attn-dir "${ATTN_DIR}" --sparsity "${SPARSITY}" --retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}")
    ;;
esac

python -m streaming.ReKV.run_eval \
  "${COMMON_ARGS[@]}" \
  ${EXTRA_ARGS}
