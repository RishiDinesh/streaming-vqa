#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"
cd "${ROOT}"
STREAMING_ENV_PREFERENCE=rocm activate_streaming_env

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

DATASET=${DATASET:-rvs_movie}
MODEL=${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
HF_REPO_ID=${HF_REPO_ID:-Becomebright/RVS}
VIDEO_INDEX=${VIDEO_INDEX:-0}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-4}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
AUDIT_ROOT=${AUDIT_ROOT:-outputs/evaluations_streaming/${DATASET//_/-}/rocm_backend_audit/video${VIDEO_INDEX}}
MODEL_SLUG=${MODEL//\//__}
FPS_SLUG=${SAMPLE_FPS//./p}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-outputs/evaluations_streaming/feature_cache/${DATASET//_/-}/${MODEL_SLUG}/fps_${FPS_SLUG}}
USE_FEATURE_CACHE=${USE_FEATURE_CACHE:-0}
DUO_STRICT_NO_SDPA_FALLBACK=${DUO_STRICT_NO_SDPA_FALLBACK:-0}
PROFILE_PROBE_FRAME_COUNTS=${PROFILE_PROBE_FRAME_COUNTS:-1,2,4,8,16}
PROFILE_QUESTION=${PROFILE_QUESTION:-What is happening in the video so far?}
EXTRA_ARGS=${EXTRA_ARGS:-}

COMMON_ARGS=(
  --dataset "${DATASET}"
  --hf-repo-id "${HF_REPO_ID}"
  --allow-hf-video-download
  --model "${MODEL}"
  --sample-fps "${SAMPLE_FPS}"
  --video-index "${VIDEO_INDEX}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --video-decode-threads "${VIDEO_DECODE_THREADS}"
)

if [[ "${USE_FEATURE_CACHE}" == "1" && -f "${FEATURE_CACHE_ROOT}/manifest.json" ]]; then
  COMMON_ARGS+=(--feature-cache-root "${FEATURE_CACHE_ROOT}")
fi

if [[ "${DUO_STRICT_NO_SDPA_FALLBACK}" == "1" ]]; then
  COMMON_ARGS+=(--duo-strict-no-sdpa-fallback)
fi

VALIDATOR_ARGS=()
if [[ "${DUO_STRICT_NO_SDPA_FALLBACK}" == "1" ]]; then
  VALIDATOR_ARGS+=(--duo-strict-no-sdpa-fallback)
fi

mkdir -p "${AUDIT_ROOT}/results" "${AUDIT_ROOT}/profiles"

python -m streaming.ReKV.validate_rocm_env \
  --output-path "${AUDIT_ROOT}/rocm_env_summary.json" \
  "${VALIDATOR_ARGS[@]}"

run_eval_one() {
  local method=$1
  shift
  python -m streaming.ReKV.run_eval \
    "${COMMON_ARGS[@]}" \
    --method "${method}" \
    --max-videos 1 \
    --max-conversations-per-video 1 \
    --output-path "${AUDIT_ROOT}/results/${method}.json" \
    "$@" \
    ${EXTRA_ARGS}
}

run_profile_one() {
  local method=$1
  shift
  python -m streaming.ReKV.profile_streaming \
    "${COMMON_ARGS[@]}" \
    --method "${method}" \
    --probe-frame-counts "${PROFILE_PROBE_FRAME_COUNTS}" \
    --profiling-question "${PROFILE_QUESTION}" \
    --output-path "${AUDIT_ROOT}/profiles/${method}.json" \
    "$@" \
    ${EXTRA_ARGS}
}

run_eval_one full_streaming
run_eval_one duo_streaming --attn-dir "${ATTN_DIR}" --sparsity 0.5
run_eval_one rekv --retrieve-size 64 --n-local 15000
run_eval_one duo_plus_rekv --attn-dir "${ATTN_DIR}" --sparsity 0.5 --retrieve-size 64 --n-local 15000

run_profile_one full_streaming
run_profile_one duo_streaming --attn-dir "${ATTN_DIR}" --sparsity 0.5
run_profile_one rekv --retrieve-size 64 --n-local 15000
run_profile_one duo_plus_rekv --attn-dir "${ATTN_DIR}" --sparsity 0.5 --retrieve-size 64 --n-local 15000

python -m streaming.ReKV.build_backend_audit_report \
  --result-dir "${AUDIT_ROOT}/results" \
  --profile-dir "${AUDIT_ROOT}/profiles" \
  --env-summary "${AUDIT_ROOT}/rocm_env_summary.json" \
  --output-path "${AUDIT_ROOT}/backend_audit_report.md"

echo "Saved ROCm backend audit outputs under ${AUDIT_ROOT}"
