#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_streaming_full_eval_local.sh <mode>

Modes:
  precompute      Build the shared visual feature cache only
  full            Run full_streaming
  duo             Run duo_streaming with sparsity=0.5
  rekv            Run rekv with configurable retrieve_size and n_local
  ab              Run duo_plus_rekv with the selected hybrid sparsity
  judge           Judge any existing promoted full-eval JSONs in place
  plots           Render plots for any existing promoted full-eval JSONs
  qualitative     Build a qualitative bundle for any existing promoted full-eval JSONs
  all             Run the official four methods from cache, then judge them in place

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
  FLUSH_EVERY_CONVERSATIONS
  ATTN_DIR
  AB_SPARSITY
  REKV_TOPK
  REKV_N_LOCAL
  AB_TOPK
  AB_N_LOCAL
  AB_DEPLOY_SINK_SIZE
  AB_DEPLOY_RECENT_SIZE
  VIDEO_DECODE_THREADS
  CLEAR_CUDA_CACHE_ON_RESET
  FEATURE_CACHE_ROOT
  FEATURE_BATCH_SIZE
  USE_FEATURE_CACHE
  RESUME
  OVERWRITE
  EXTRA_ARGS

Examples:
  DATASET=rvs_ego scripts/run_streaming_full_eval_local.sh precompute
  DATASET=rvs_ego scripts/run_streaming_full_eval_local.sh all
  DATASET=rvs_movie RESUME=1 scripts/run_streaming_full_eval_local.sh ab
  DATASET=rvs_movie scripts/run_streaming_full_eval_local.sh judge
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
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/evaluations_streaming/${DATASET//_/-}/full_eval}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
FPS_SLUG=${SAMPLE_FPS//./p}
MODEL_SLUG=${MODEL//\//__}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-outputs/evaluations_streaming/feature_cache/${DATASET//_/-}/${MODEL_SLUG}/fps_${FPS_SLUG}}
FEATURE_BATCH_SIZE=${FEATURE_BATCH_SIZE:-16}
USE_FEATURE_CACHE=${USE_FEATURE_CACHE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
MAX_VIDEOS=${MAX_VIDEOS:-}
MAX_CONVERSATIONS=${MAX_CONVERSATIONS:-}
VIDEO_OFFSET=${VIDEO_OFFSET:-0}
FLUSH_EVERY_VIDEOS=${FLUSH_EVERY_VIDEOS:-1}
FLUSH_EVERY_CONVERSATIONS=${FLUSH_EVERY_CONVERSATIONS:-1}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
AB_SPARSITY=${AB_SPARSITY:-0.5}
REKV_TOPK=${REKV_TOPK:-64}
REKV_N_LOCAL=${REKV_N_LOCAL:-15000}
AB_TOPK=${AB_TOPK:-${REKV_TOPK}}
AB_N_LOCAL=${AB_N_LOCAL:-${REKV_N_LOCAL}}
AB_DEPLOY_SINK_SIZE=${AB_DEPLOY_SINK_SIZE:-}
AB_DEPLOY_RECENT_SIZE=${AB_DEPLOY_RECENT_SIZE:-}
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-4}
CLEAR_CUDA_CACHE_ON_RESET=${CLEAR_CUDA_CACHE_ON_RESET:-0}
RESUME=${RESUME:-1}
OVERWRITE=${OVERWRITE:-0}
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
  --flush-every-videos "${FLUSH_EVERY_VIDEOS}"
  --flush-every-conversations "${FLUSH_EVERY_CONVERSATIONS}"
  --video-decode-threads "${VIDEO_DECODE_THREADS}"
)

if [[ "${CLEAR_CUDA_CACHE_ON_RESET}" == "1" ]]; then
  COMMON_ARGS+=(--clear-cuda-cache-on-reset)
fi

if [[ -n "${MAX_VIDEOS}" ]]; then
  COMMON_ARGS+=(--max-videos "${MAX_VIDEOS}")
fi

if [[ -n "${MAX_CONVERSATIONS}" ]]; then
  COMMON_ARGS+=(--max-conversations-per-video "${MAX_CONVERSATIONS}")
fi

ensure_feature_cache_requested() {
  if [[ "${USE_FEATURE_CACHE}" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "${FEATURE_CACHE_ROOT}/manifest.json" ]]; then
    echo "Feature cache manifest not found under ${FEATURE_CACHE_ROOT}" >&2
    echo "Run: DATASET=${DATASET} scripts/run_streaming_full_eval_local.sh precompute" >&2
    exit 1
  fi
}

format_sparsity_tag() {
  local value=$1
  printf 's%s' "${value//./}"
}

build_rekv_tag() {
  printf 'rekv_topk%s_nlocal%s' "${REKV_TOPK}" "${REKV_N_LOCAL}"
}

build_ab_tag() {
  local tag="duo_plus_rekv_selected_$(format_sparsity_tag "${AB_SPARSITY}")"
  if [[ -n "${AB_DEPLOY_SINK_SIZE}" ]]; then
    tag+="_sink${AB_DEPLOY_SINK_SIZE}"
  fi
  if [[ -n "${AB_DEPLOY_RECENT_SIZE}" ]]; then
    tag+="_recent${AB_DEPLOY_RECENT_SIZE}"
  fi
  tag+="_topk${AB_TOPK}_nlocal${AB_N_LOCAL}"
  printf '%s' "${tag}"
}

declare -a AB_DEPLOY_ARGS=()
if [[ -n "${AB_DEPLOY_SINK_SIZE}" ]]; then
  AB_DEPLOY_ARGS+=(--deploy-sink-size "${AB_DEPLOY_SINK_SIZE}")
fi
if [[ -n "${AB_DEPLOY_RECENT_SIZE}" ]]; then
  AB_DEPLOY_ARGS+=(--deploy-recent-size "${AB_DEPLOY_RECENT_SIZE}")
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
  if [[ "${USE_FEATURE_CACHE}" == "1" ]]; then
    args+=(--feature-cache-root "${FEATURE_CACHE_ROOT}")
  fi
  python -m streaming.ReKV.run_eval \
    "${args[@]}" \
    --method "${method}" \
    --output-path "${output_path}" \
    "$@" \
    ${EXTRA_ARGS}
}

run_precompute() {
  local args=(
    --dataset "${DATASET}"
    --hf-repo-id "${HF_REPO_ID}"
    --allow-hf-video-download
    --model "${MODEL}"
    --sample-fps "${SAMPLE_FPS}"
    --video-offset "${VIDEO_OFFSET}"
    --feature-batch-size "${FEATURE_BATCH_SIZE}"
    --video-decode-threads "${VIDEO_DECODE_THREADS}"
    --feature-cache-root "${FEATURE_CACHE_ROOT}"
  )
  if [[ -n "${MAX_VIDEOS}" ]]; then
    args+=(--max-videos "${MAX_VIDEOS}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite-existing)
  fi
  python -m streaming.ReKV.precompute_features \
    "${args[@]}" \
    ${EXTRA_ARGS}
}

judge_all() {
  local files=()
  local method_dir
  for method_dir in full_streaming duo_streaming rekv duo_plus_rekv; do
    if [[ -d "${OUTPUT_ROOT}/${method_dir}" ]]; then
      while IFS= read -r path; do
        files+=("${path}")
      done < <(find "${OUTPUT_ROOT}/${method_dir}" -maxdepth 1 -type f -name '*.json' | sort)
    fi
  done
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No full-eval JSONs found under ${OUTPUT_ROOT}" >&2
    exit 1
  fi
  python -m streaming.ReKV.judge_results "${files[@]}" --in-place
}

plot_all() {
  local files=()
  local method_dir
  for method_dir in full_streaming duo_streaming rekv duo_plus_rekv; do
    if [[ -d "${OUTPUT_ROOT}/${method_dir}" ]]; then
      while IFS= read -r path; do
        files+=("${path}")
      done < <(find "${OUTPUT_ROOT}/${method_dir}" -maxdepth 1 -type f -name '*.json' | sort)
    fi
  done
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No full-eval JSONs found under ${OUTPUT_ROOT}" >&2
    exit 1
  fi
  python -m streaming.ReKV.plot_results "${files[@]}" --output-dir "${OUTPUT_ROOT}/plots"
}

qualitative_all() {
  local files=()
  local method_dir
  for method_dir in full_streaming duo_streaming rekv duo_plus_rekv; do
    if [[ -d "${OUTPUT_ROOT}/${method_dir}" ]]; then
      while IFS= read -r path; do
        files+=("${path}")
      done < <(find "${OUTPUT_ROOT}/${method_dir}" -maxdepth 1 -type f -name '*.json' | sort)
    fi
  done
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No full-eval JSONs found under ${OUTPUT_ROOT}" >&2
    exit 1
  fi
  python -m streaming.ReKV.build_qualitative_bundle \
    "${files[@]}" \
    --output-dir "${OUTPUT_ROOT}/qualitative"
}

case "${MODE}" in
  precompute)
    run_precompute
    ;;
  full)
    ensure_feature_cache_requested
    run_one full_streaming full_streaming
    ;;
  duo)
    ensure_feature_cache_requested
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    ;;
  rekv)
    ensure_feature_cache_requested
    run_one rekv "$(build_rekv_tag)" --retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}"
    ;;
  ab)
    ensure_feature_cache_requested
    run_one duo_plus_rekv "$(build_ab_tag)" \
      --attn-dir "${ATTN_DIR}" \
      --sparsity "${AB_SPARSITY}" \
      --retrieve-size "${AB_TOPK}" \
      --n-local "${AB_N_LOCAL}" \
      "${AB_DEPLOY_ARGS[@]}"
    ;;
  judge)
    judge_all
    ;;
  plots)
    plot_all
    ;;
  qualitative)
    qualitative_all
    ;;
  all)
    ensure_feature_cache_requested
    run_one full_streaming full_streaming
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    run_one rekv "$(build_rekv_tag)" --retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}"
    run_one duo_plus_rekv "$(build_ab_tag)" \
      --attn-dir "${ATTN_DIR}" \
      --sparsity "${AB_SPARSITY}" \
      --retrieve-size "${AB_TOPK}" \
      --n-local "${AB_N_LOCAL}" \
      "${AB_DEPLOY_ARGS[@]}"
    judge_all
    plot_all
    qualitative_all
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
