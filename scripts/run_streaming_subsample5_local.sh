#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_streaming_subsample5_local.sh <mode>

Modes:
  full            Run full_streaming on the standard subsample5 slice
  duo             Run duo_streaming with sparsity=0.5
  duo_fullheads   Run duo_streaming with sparsity=0.0
  rekv            Run rekv with retrieve_size=64 and n_local=15000
  ab              Run duo_plus_rekv with sparsity=0.5 and ReKV defaults
  all             Run all five sequentially

Environment overrides:
  DATASET
  MODEL
  HF_REPO_ID
  SUBSAMPLE_NAME
  MAX_VIDEOS
  MAX_CONVERSATIONS
  SAMPLE_FPS
  MAX_NEW_TOKENS
  VIDEO_OFFSET
  ATTN_DIR
  EXTRA_ARGS

Examples:
  scripts/run_streaming_subsample5_local.sh duo
  scripts/run_streaming_subsample5_local.sh all
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 1
fi

MODE=$1
ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET=${DATASET:-rvs_ego}
MODEL=${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
HF_REPO_ID=${HF_REPO_ID:-Becomebright/RVS}
SUBSAMPLE_NAME=${SUBSAMPLE_NAME:-subsample5}
MAX_VIDEOS=${MAX_VIDEOS:-5}
MAX_CONVERSATIONS=${MAX_CONVERSATIONS:-3}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
VIDEO_OFFSET=${VIDEO_OFFSET:-0}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
EXTRA_ARGS=${EXTRA_ARGS:-}

source /root/miniforge3/etc/profile.d/conda.sh
conda activate duo

cd "${ROOT}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --hf-repo-id "${HF_REPO_ID}"
  --allow-hf-video-download
  --model "${MODEL}"
  --sample-fps "${SAMPLE_FPS}"
  --max-videos "${MAX_VIDEOS}"
  --video-offset "${VIDEO_OFFSET}"
  --max-conversations-per-video "${MAX_CONVERSATIONS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --subsample-name "${SUBSAMPLE_NAME}"
  --flush-every-videos 1
  --overwrite-output
)

OUT_ROOT="outputs/evaluations_streaming/${DATASET//_/-}/${SUBSAMPLE_NAME}"

run_one() {
  local method=$1
  local tag=$2
  shift 2
  local output_path="${OUT_ROOT}/${method}/${tag}.json"
  python -m streaming.ReKV.run_eval \
    "${COMMON_ARGS[@]}" \
    --method "${method}" \
    --output-path "${output_path}" \
    "$@" \
    ${EXTRA_ARGS}
}

case "${MODE}" in
  full)
    run_one full_streaming full_streaming
    ;;
  duo)
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    ;;
  duo_fullheads)
    run_one duo_streaming duo_streaming_s00 --attn-dir "${ATTN_DIR}" --sparsity 0.0
    ;;
  rekv)
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    ;;
  ab)
    run_one duo_plus_rekv duo_plus_rekv_s05_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.5 \
      --retrieve-size 64 \
      --n-local 15000
    ;;
  all)
    run_one full_streaming full_streaming
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    run_one duo_streaming duo_streaming_s00 --attn-dir "${ATTN_DIR}" --sparsity 0.0
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    run_one duo_plus_rekv duo_plus_rekv_s05_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.5 \
      --retrieve-size 64 \
      --n-local 15000
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
