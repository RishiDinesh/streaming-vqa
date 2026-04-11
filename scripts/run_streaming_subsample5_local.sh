#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "${ROOT}/scripts/streaming_env.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_streaming_subsample5_local.sh <mode>

Modes:
  full            Run full_streaming on the standard subsample5 slice
  duo             Run duo_streaming with sparsity=0.5
  rekv            Run rekv with configurable retrieve_size and n_local
  ab_s05          Run duo_plus_rekv with sparsity=0.5
  ab_s075         Run duo_plus_rekv with sparsity=0.75
  judge           Judge any existing method JSONs in place
  plots           Render all available plots for the current slice
  qualitative     Build a qualitative bundle for the current slice
  all             Run all methods, judge, plot, and build the qualitative bundle

Environment overrides:
  DATASET
  MODEL
  HF_REPO_ID
  SUBSAMPLE_NAME
  FEATURE_CACHE_ROOT
  USE_FEATURE_CACHE
  MAX_VIDEOS
  MAX_CONVERSATIONS
  SAMPLE_FPS
  MAX_NEW_TOKENS
  VIDEO_OFFSET
  FLUSH_EVERY_CONVERSATIONS
  ATTN_DIR
  OUTPUT_ROOT
  VIDEO_DECODE_THREADS
  CLEAR_CUDA_CACHE_ON_RESET
  REKV_TOPK
  REKV_N_LOCAL
  AB_TOPK
  AB_N_LOCAL
  AB_DEPLOY_SINK_SIZE
  AB_DEPLOY_RECENT_SIZE
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
DATASET=${DATASET:-rvs_ego}
MODEL=${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
HF_REPO_ID=${HF_REPO_ID:-Becomebright/RVS}
SUBSAMPLE_NAME=${SUBSAMPLE_NAME:-subsample5}
MODEL_SLUG=${MODEL//\//__}
FPS_SLUG=${SAMPLE_FPS:-0.5}
FPS_SLUG=${FPS_SLUG//./p}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-outputs/evaluations_streaming/feature_cache/${DATASET//_/-}/${MODEL_SLUG}/fps_${FPS_SLUG}}
USE_FEATURE_CACHE=${USE_FEATURE_CACHE:-0}
MAX_VIDEOS=${MAX_VIDEOS:-5}
MAX_CONVERSATIONS=${MAX_CONVERSATIONS:-3}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
VIDEO_OFFSET=${VIDEO_OFFSET:-0}
FLUSH_EVERY_CONVERSATIONS=${FLUSH_EVERY_CONVERSATIONS:-1}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-4}
CLEAR_CUDA_CACHE_ON_RESET=${CLEAR_CUDA_CACHE_ON_RESET:-0}
REKV_TOPK=${REKV_TOPK:-64}
REKV_N_LOCAL=${REKV_N_LOCAL:-15000}
AB_TOPK=${AB_TOPK:-${REKV_TOPK}}
AB_N_LOCAL=${AB_N_LOCAL:-${REKV_N_LOCAL}}
AB_DEPLOY_SINK_SIZE=${AB_DEPLOY_SINK_SIZE:-}
AB_DEPLOY_RECENT_SIZE=${AB_DEPLOY_RECENT_SIZE:-}
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
  --max-videos "${MAX_VIDEOS}"
  --video-offset "${VIDEO_OFFSET}"
  --max-conversations-per-video "${MAX_CONVERSATIONS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --subsample-name "${SUBSAMPLE_NAME}"
  --flush-every-videos 1
  --flush-every-conversations "${FLUSH_EVERY_CONVERSATIONS}"
  --video-decode-threads "${VIDEO_DECODE_THREADS}"
  --overwrite-output
)

if [[ "${USE_FEATURE_CACHE}" == "1" ]]; then
  if [[ ! -f "${FEATURE_CACHE_ROOT}/manifest.json" ]]; then
    echo "Feature cache manifest not found under ${FEATURE_CACHE_ROOT}" >&2
    echo "Build it first, for example:" >&2
    echo "  DATASET=${DATASET} MODEL=${MODEL} bash scripts/run_streaming_full_eval_local.sh precompute" >&2
    exit 1
  fi
  COMMON_ARGS+=(--feature-cache-root "${FEATURE_CACHE_ROOT}")
fi

if [[ "${CLEAR_CUDA_CACHE_ON_RESET}" == "1" ]]; then
  COMMON_ARGS+=(--clear-cuda-cache-on-reset)
fi

OUT_ROOT="outputs/evaluations_streaming/${DATASET//_/-}/${SUBSAMPLE_NAME}"
OUT_ROOT=${OUTPUT_ROOT:-${OUT_ROOT}}

format_sparsity_tag() {
  local value=$1
  printf 's%s' "${value//./}"
}

build_rekv_tag() {
  printf 'rekv_topk%s_nlocal%s' "${REKV_TOPK}" "${REKV_N_LOCAL}"
}

build_ab_tag() {
  local sparsity=$1
  local tag="duo_plus_rekv_$(format_sparsity_tag "${sparsity}")"
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
  local output_path="${OUT_ROOT}/${method}/${tag}.json"
  python -m streaming.ReKV.run_eval \
    "${COMMON_ARGS[@]}" \
    --method "${method}" \
    --output-path "${output_path}" \
    "$@" \
    ${EXTRA_ARGS}
}

collect_existing_files() {
  local files=()
  local method_dir
  for method_dir in full_streaming duo_streaming rekv duo_plus_rekv; do
    if [[ -d "${OUT_ROOT}/${method_dir}" ]]; then
      while IFS= read -r path; do
        files+=("${path}")
      done < <(find "${OUT_ROOT}/${method_dir}" -maxdepth 1 -type f -name '*.json' | sort)
    fi
  done
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No result JSONs found under ${OUT_ROOT}" >&2
    exit 1
  fi
  printf '%s\n' "${files[@]}"
}

judge_all() {
  mapfile -t files < <(collect_existing_files)
  python -m streaming.ReKV.judge_results "${files[@]}" --in-place
}

plot_all() {
  mapfile -t files < <(collect_existing_files)
  python -m streaming.ReKV.plot_results "${files[@]}" --output-dir "${OUT_ROOT}/plots"
}

qualitative_all() {
  mapfile -t files < <(collect_existing_files)
  python -m streaming.ReKV.build_qualitative_bundle \
    "${files[@]}" \
    --output-dir "${OUT_ROOT}/qualitative"
}

case "${MODE}" in
  full)
    run_one full_streaming full_streaming
    ;;
  duo)
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    ;;
  rekv)
    run_one rekv "$(build_rekv_tag)" --retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}"
    ;;
  ab_s05)
    run_one duo_plus_rekv "$(build_ab_tag 0.5)" \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.5 \
      --retrieve-size "${AB_TOPK}" \
      --n-local "${AB_N_LOCAL}" \
      "${AB_DEPLOY_ARGS[@]}"
    ;;
  ab_s075)
    run_one duo_plus_rekv "$(build_ab_tag 0.75)" \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.75 \
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
    run_one full_streaming full_streaming
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    run_one rekv "$(build_rekv_tag)" --retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}"
    run_one duo_plus_rekv "$(build_ab_tag 0.5)" \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.5 \
      --retrieve-size "${AB_TOPK}" \
      --n-local "${AB_N_LOCAL}" \
      "${AB_DEPLOY_ARGS[@]}"
    run_one duo_plus_rekv "$(build_ab_tag 0.75)" \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.75 \
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
