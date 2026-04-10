#!/usr/bin/env bash
set -euo pipefail

activate_duo_env() {
  # Prefer /opt/venv (torch 2.9 + ROCm 7) over the duo conda env (torch 2.4 + ROCm 6.1).
  # ROCm 7 ships tuned MI300X kernels that are ~3-5x faster for this model.
  local _check_pkgs='
import importlib
for name in ("torch", "numpy", "matplotlib", "transformers", "tqdm", "duo_attn"):
    importlib.import_module(name)
'

  if [[ -x "/opt/venv/bin/python" ]]; then
    if /opt/venv/bin/python - <<PY >/dev/null 2>&1
${_check_pkgs}
PY
    then
      export PATH="/opt/venv/bin:${PATH}"
      echo "[env] Using /opt/venv (torch $(/opt/venv/bin/python -c 'import torch; print(torch.__version__)'))" >&2
      return
    fi
  fi

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
    echo "[warn] Could not locate preferred environment; using current Python: $(command -v python)" >&2
    return
  fi

  echo "Could not locate a suitable Python environment with required packages." >&2
  exit 1
}

usage() {
    cat <<'EOF'
Usage:
  scripts/run_streaming_subsample5_local.sh <mode>

Modes:
  full            Run full_streaming on the standard subsample5 slice
  duo             Run duo_streaming with sparsity=0.5
  rekv            Run rekv with retrieve_size=64 and n_local=15000
  rekv_no_offload Run the short-memory ReKV ablation without long-range offload
  ab_s0375        Run duo_plus_rekv with sparsity=0.375
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
FLUSH_EVERY_CONVERSATIONS=${FLUSH_EVERY_CONVERSATIONS:-1}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-4}
CLEAR_CUDA_CACHE_ON_RESET=${CLEAR_CUDA_CACHE_ON_RESET:-0}
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

if [[ "${CLEAR_CUDA_CACHE_ON_RESET}" == "1" ]]; then
  COMMON_ARGS+=(--clear-cuda-cache-on-reset)
fi

OUT_ROOT="outputs/evaluations_streaming/${DATASET//_/-}/${SUBSAMPLE_NAME}"
OUT_ROOT=${OUTPUT_ROOT:-${OUT_ROOT}}

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
  local candidates=(
    "${OUT_ROOT}/full_streaming/full_streaming.json"
    "${OUT_ROOT}/duo_streaming/duo_streaming_s05.json"
    "${OUT_ROOT}/rekv/rekv_topk64_nlocal15000.json"
    "${OUT_ROOT}/rekv_no_offload/rekv_no_offload_nlocal15000.json"
    "${OUT_ROOT}/duo_plus_rekv/duo_plus_rekv_s0375_topk64_nlocal15000.json"
    "${OUT_ROOT}/duo_plus_rekv/duo_plus_rekv_s05_topk64_nlocal15000.json"
    "${OUT_ROOT}/duo_plus_rekv/duo_plus_rekv_s075_topk64_nlocal15000.json"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -f "${path}" ]]; then
      files+=("${path}")
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
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    ;;
  rekv_no_offload)
    run_one rekv_no_offload rekv_no_offload_nlocal15000 --n-local 15000
    ;;
  ab_s0375)
    run_one duo_plus_rekv duo_plus_rekv_s0375_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.375 \
      --retrieve-size 64 \
      --n-local 15000
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
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    run_one rekv_no_offload rekv_no_offload_nlocal15000 --n-local 15000
    run_one duo_plus_rekv duo_plus_rekv_s0375_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.375 \
      --retrieve-size 64 \
      --n-local 15000
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
    judge_all
    plot_all
    qualitative_all
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
