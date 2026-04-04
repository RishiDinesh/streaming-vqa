#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_streaming_full_eval_local.sh <mode>

Modes:
  full            Run full_streaming
  duo             Run duo_streaming with sparsity=0.5
  duo_fullheads   Run duo_streaming with sparsity=0.0
  rekv            Run rekv with retrieve_size=64 and n_local=15000
  ab              Run duo_plus_rekv with sparsity=0.375 and ReKV defaults
  judge           Judge any existing promoted full-eval JSONs in place
  all             Run all five methods sequentially, then judge them in place

Environment overrides:
  DATASET
  MODEL
  HF_REPO_ID
  OUTPUT_ROOT
  SAMPLE_FPS
  MAX_NEW_TOKENS
  MAX_VIDEOS
  MAX_CONVERSATIONS
  VIDEO_OFFSET
  FLUSH_EVERY_VIDEOS
  ATTN_DIR
  RESUME
  OVERWRITE
  EXTRA_ARGS

Examples:
  DATASET=rvs_ego scripts/run_streaming_full_eval_local.sh all
  DATASET=rvs_movie RESUME=1 scripts/run_streaming_full_eval_local.sh rekv
  DATASET=rvs_movie scripts/run_streaming_full_eval_local.sh judge
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
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/evaluations_streaming/${DATASET//_/-}/full_eval}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
MAX_VIDEOS=${MAX_VIDEOS:-}
MAX_CONVERSATIONS=${MAX_CONVERSATIONS:-}
VIDEO_OFFSET=${VIDEO_OFFSET:-0}
FLUSH_EVERY_VIDEOS=${FLUSH_EVERY_VIDEOS:-1}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
RESUME=${RESUME:-1}
OVERWRITE=${OVERWRITE:-0}
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
  --video-offset "${VIDEO_OFFSET}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --flush-every-videos "${FLUSH_EVERY_VIDEOS}"
)

if [[ -n "${MAX_VIDEOS}" ]]; then
  COMMON_ARGS+=(--max-videos "${MAX_VIDEOS}")
fi

if [[ -n "${MAX_CONVERSATIONS}" ]]; then
  COMMON_ARGS+=(--max-conversations-per-video "${MAX_CONVERSATIONS}")
fi

run_one() {
  local method=$1
  local tag=$2
  shift 2
  local output_path="${OUTPUT_ROOT}/${method}/${tag}.json"
  local args=("${COMMON_ARGS[@]}")
  if [[ -f "${output_path}" && "${RESUME}" == "1" ]]; then
    args+=(--resume)
  elif [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite-output)
  fi
  python -m streaming.ReKV.run_eval \
    "${args[@]}" \
    --method "${method}" \
    --output-path "${output_path}" \
    "$@" \
    ${EXTRA_ARGS}
}

judge_all() {
  local files=()
  local maybe_files=(
    "${OUTPUT_ROOT}/full_streaming/full_streaming.json"
    "${OUTPUT_ROOT}/duo_streaming/duo_streaming_s05.json"
    "${OUTPUT_ROOT}/duo_streaming/duo_streaming_s00.json"
    "${OUTPUT_ROOT}/rekv/rekv_topk64_nlocal15000.json"
    "${OUTPUT_ROOT}/duo_plus_rekv/duo_plus_rekv_s0375_topk64_nlocal15000.json"
  )
  local path
  for path in "${maybe_files[@]}"; do
    if [[ -f "${path}" ]]; then
      files+=("${path}")
    fi
  done
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No full-eval JSONs found under ${OUTPUT_ROOT}" >&2
    exit 1
  fi
  python -m streaming.ReKV.judge_results "${files[@]}" --in-place
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
    run_one duo_plus_rekv duo_plus_rekv_s0375_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.375 \
      --retrieve-size 64 \
      --n-local 15000
    ;;
  judge)
    judge_all
    ;;
  all)
    run_one full_streaming full_streaming
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    run_one duo_streaming duo_streaming_s00 --attn-dir "${ATTN_DIR}" --sparsity 0.0
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    run_one duo_plus_rekv duo_plus_rekv_s0375_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.375 \
      --retrieve-size 64 \
      --n-local 15000
    judge_all
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
