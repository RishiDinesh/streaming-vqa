#!/usr/bin/env bash
set -euo pipefail

activate_duo_env() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate duo
    return
  fi

  local candidate
  for candidate in \
    "/root/miniforge3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/usr/local/miniconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      conda activate duo
      return
    fi
  done

  if python - <<'PY' >/dev/null 2>&1
import importlib
for name in ("torch", "numpy", "matplotlib", "transformers", "tqdm"):
    importlib.import_module(name)
PY
  then
    echo "[warn] Could not locate the 'duo' Conda environment; using current Python: $(command -v python)" >&2
    return
  fi

  echo "Could not locate Conda activation script for the 'duo' environment, and the current Python is missing required packages." >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  scripts/run_streaming_full_eval_local.sh <mode>

Modes:
  precompute      Build the shared visual feature cache only
  full            Run full_streaming
  duo             Run duo_streaming with sparsity=0.5
  rekv            Run rekv with retrieve_size=64 and n_local=15000
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
ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
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
AB_SPARSITY=${AB_SPARSITY:-0.375}
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-4}
CLEAR_CUDA_CACHE_ON_RESET=${CLEAR_CUDA_CACHE_ON_RESET:-0}
RESUME=${RESUME:-1}
OVERWRITE=${OVERWRITE:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

activate_duo_env

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
  local maybe_files=(
    "${OUTPUT_ROOT}/full_streaming/full_streaming.json"
    "${OUTPUT_ROOT}/duo_streaming/duo_streaming_s05.json"
    "${OUTPUT_ROOT}/rekv/rekv_topk64_nlocal15000.json"
    "${OUTPUT_ROOT}/duo_plus_rekv/duo_plus_rekv_selected_topk64_nlocal15000.json"
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

plot_all() {
  local files=()
  local maybe_files=(
    "${OUTPUT_ROOT}/full_streaming/full_streaming.json"
    "${OUTPUT_ROOT}/duo_streaming/duo_streaming_s05.json"
    "${OUTPUT_ROOT}/rekv/rekv_topk64_nlocal15000.json"
    "${OUTPUT_ROOT}/duo_plus_rekv/duo_plus_rekv_selected_topk64_nlocal15000.json"
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
  python -m streaming.ReKV.plot_results "${files[@]}" --output-dir "${OUTPUT_ROOT}/plots"
}

qualitative_all() {
  local files=()
  local maybe_files=(
    "${OUTPUT_ROOT}/full_streaming/full_streaming.json"
    "${OUTPUT_ROOT}/duo_streaming/duo_streaming_s05.json"
    "${OUTPUT_ROOT}/rekv/rekv_topk64_nlocal15000.json"
    "${OUTPUT_ROOT}/duo_plus_rekv/duo_plus_rekv_selected_topk64_nlocal15000.json"
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
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    ;;
  ab)
    ensure_feature_cache_requested
    run_one duo_plus_rekv duo_plus_rekv_selected_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity "${AB_SPARSITY}" \
      --retrieve-size 64 \
      --n-local 15000
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
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    run_one duo_plus_rekv duo_plus_rekv_selected_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity "${AB_SPARSITY}" \
      --retrieve-size 64 \
      --n-local 15000
    judge_all
    plot_all
    qualitative_all
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
