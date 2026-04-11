#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_streaming_profile_local.sh <mode>

Modes:
  full            Profile full_streaming on one representative video
  duo             Profile duo_streaming with sparsity=0.5
  rekv            Profile rekv
  ab_s05          Profile duo_plus_rekv with sparsity=0.5
  ab_s075         Profile duo_plus_rekv with sparsity=0.75
  all             Run the recommended profiling set
  plots           Render profile plots for existing JSONs

Environment overrides:
  DATASET
  MODEL
  HF_REPO_ID
  VIDEO_INDEX
  VIDEO_ID
  VIDEO_OFFSET
  SAMPLE_FPS
  MAX_NEW_TOKENS
  ATTN_DIR
  FEATURE_CACHE_ROOT
  USE_FEATURE_CACHE
  PROFILE_OUTPUT_ROOT
  PROFILE_NAME
  PROFILE_QUESTION
  PROBE_FRAME_COUNTS
  VIDEO_DECODE_THREADS
  CLEAR_CUDA_CACHE_ON_RESET
  EXTRA_ARGS
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

MODE=$1
DATASET=${DATASET:-rvs_ego}
MODEL=${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
HF_REPO_ID=${HF_REPO_ID:-Becomebright/RVS}
VIDEO_INDEX=${VIDEO_INDEX:-0}
VIDEO_ID=${VIDEO_ID:-}
VIDEO_OFFSET=${VIDEO_OFFSET:-0}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
MODEL_SLUG=${MODEL//\//__}
FPS_SLUG=${SAMPLE_FPS//./p}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-outputs/evaluations_streaming/feature_cache/${DATASET//_/-}/${MODEL_SLUG}/fps_${FPS_SLUG}}
USE_FEATURE_CACHE=${USE_FEATURE_CACHE:-0}
PROFILE_OUTPUT_ROOT=${PROFILE_OUTPUT_ROOT:-outputs/evaluations_streaming/${DATASET//_/-}/profiles}
PROFILE_NAME=${PROFILE_NAME:-profile_video${VIDEO_INDEX}}
PROFILE_QUESTION="${PROFILE_QUESTION:-What is happening in the video so far?}"
PROBE_FRAME_COUNTS=${PROBE_FRAME_COUNTS:-1,2,4,8,16,32,64,128}
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-4}
CLEAR_CUDA_CACHE_ON_RESET=${CLEAR_CUDA_CACHE_ON_RESET:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

activate_streaming_env

cd "${ROOT}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

COMMON_ARGS=(
  --dataset "${DATASET}"
  --hf-repo-id "${HF_REPO_ID}"
  --allow-hf-video-download
  --model "${MODEL}"
  --sample-fps "${SAMPLE_FPS}"
  --video-offset "${VIDEO_OFFSET}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --probe-frame-counts "${PROBE_FRAME_COUNTS}"
  --profiling-question "${PROFILE_QUESTION}"
  --video-decode-threads "${VIDEO_DECODE_THREADS}"
)

if [[ "${USE_FEATURE_CACHE}" == "1" ]]; then
  if [[ ! -f "${FEATURE_CACHE_ROOT}/manifest.json" ]]; then
    echo "Feature cache manifest not found under ${FEATURE_CACHE_ROOT}" >&2
    exit 1
  fi
  COMMON_ARGS+=(--feature-cache-root "${FEATURE_CACHE_ROOT}")
fi

if [[ "${CLEAR_CUDA_CACHE_ON_RESET}" == "1" ]]; then
  COMMON_ARGS+=(--clear-cuda-cache-on-reset)
fi

if [[ -n "${VIDEO_ID}" ]]; then
  COMMON_ARGS+=(--video-id "${VIDEO_ID}")
else
  COMMON_ARGS+=(--video-index "${VIDEO_INDEX}")
fi

run_one() {
  local method=$1
  local tag=$2
  shift 2
  python -m streaming.ReKV.profile_streaming \
    "${COMMON_ARGS[@]}" \
    --method "${method}" \
    --output-path "${PROFILE_OUTPUT_ROOT}/${PROFILE_NAME}/${tag}.json" \
    "$@" \
    ${EXTRA_ARGS}
}

plot_all() {
  local profile_dir="${PROFILE_OUTPUT_ROOT}/${PROFILE_NAME}"
  if [[ ! -d "${profile_dir}" ]]; then
    echo "Profile directory not found: ${profile_dir}" >&2
    exit 1
  fi
  mapfile -t files < <(find "${profile_dir}" -maxdepth 1 -type f -name '*.json' | sort)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No profile JSONs found under ${profile_dir}" >&2
    exit 1
  fi
  python -m streaming.ReKV.plot_profile "${files[@]}" --output-dir "${profile_dir}/plots"
}

case "${MODE}" in
  full)
    run_one full_streaming full_streaming
    ;;
  duo)
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    ;;
  rekv)
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    ;;
  ab_s05)
    run_one duo_plus_rekv duo_plus_rekv_s05_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.5 \
      --retrieve-size 64 \
      --n-local 15000
    ;;
  ab_s075)
    run_one duo_plus_rekv duo_plus_rekv_s075_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.75 \
      --retrieve-size 64 \
      --n-local 15000
    ;;
  all)
    run_one full_streaming full_streaming
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    run_one duo_plus_rekv duo_plus_rekv_s05_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.5 \
      --retrieve-size 64 \
      --n-local 15000
    run_one duo_plus_rekv duo_plus_rekv_s075_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.75 \
      --retrieve-size 64 \
      --n-local 15000
    plot_all
    ;;
  plots)
    plot_all
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
