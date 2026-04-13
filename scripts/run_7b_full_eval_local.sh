#!/usr/bin/env bash
# Full eval: 7B model, all 4 methods in parallel (1 GPU each),
# sink=256/recent=512, s=0.75, both datasets (rvs_ego, rvs_movie).
#
# Flow per dataset:
#   1. Launch 4 methods in parallel (gpu0=full_streaming, gpu1=rekv,
#      gpu2=duo_streaming, gpu3=duo_plus_rekv)
#   2. Wait for all 4 to finish
#   3. Run LLM judge (in-place, adds judge_score to JSONs)
#   4. Generate plots (auto-uses judge_score)
#   5. Generate comparison summary.md
#   Then repeat for next dataset.
#
# Run from repo root:
#   bash scripts/run_7b_full_eval_local.sh 2>&1 | tee logs/7b_full_eval.log
#
# Overrides:
#   DATASETS    space-separated (default: "rvs_ego rvs_movie")
#   OUTPUT_ROOT (default: outputs/evaluations_streaming/7b_full_eval)
#   MAX_VIDEOS  limit videos per method (unset = all)
#   RESUME      resume from existing output (default: 1)
#   EXTRA_ARGS  extra args forwarded to run_eval

set -uo pipefail   # no -e: we handle errors manually around wait
ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

mkdir -p "${ROOT}/logs"

PYTHON=/opt/venv/bin/python
MODEL=llava-hf/llava-onevision-qwen2-7b-ov-hf
ATTN_DIR=outputs/train/7b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5
SPARSITY=0.75
SINK=256
RECENT=512
REKV_TOPK=64
REKV_N_LOCAL=15000
DATASETS=${DATASETS:-"rvs_ego rvs_movie"}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/evaluations_streaming/7b_full_eval}
MAX_VIDEOS=${MAX_VIDEOS:-}
RESUME=${RESUME:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${ROOT}/.hf_cache"

run_method_bg() {
  local gpu=$1 dataset=$2 method=$3 tag=$4
  shift 4
  local out="${OUTPUT_ROOT}/${dataset}/${method}/${tag}.json"
  local log="${OUTPUT_ROOT}/${dataset}/${method}/${tag}.log"
  mkdir -p "$(dirname "${out}")"

  local resume_flag=()
  local overwrite_flag=()
  if [[ -f "${out}" && "${RESUME}" == "1" ]]; then
    resume_flag=(--resume)
  else
    overwrite_flag=(--overwrite-output)
  fi

  local maxv_flag=()
  if [[ -n "${MAX_VIDEOS}" ]]; then
    maxv_flag=(--max-videos "${MAX_VIDEOS}")
  fi

  echo "[gpu${gpu}] starting ${method} → ${log}"
  env CUDA_VISIBLE_DEVICES="${gpu}" \
    $PYTHON -m streaming.ReKV.run_eval \
      --dataset "${dataset}" \
      --allow-hf-video-download \
      --model "${MODEL}" \
      --sample-fps 0.5 \
      --max-new-tokens 64 \
      --video-decode-threads 1 \
      --clear-cuda-cache-on-reset \
      --method "${method}" \
      --output-path "${out}" \
      "${resume_flag[@]}" \
      "${overwrite_flag[@]}" \
      "${maxv_flag[@]}" \
      "$@" \
      ${EXTRA_ARGS} \
      > "${log}" 2>&1 &
  echo $! > "${log%.log}.pid"
}

wait_all() {
  local dataset=$1
  local pid_files=(
    "${OUTPUT_ROOT}/${dataset}/full_streaming/full_streaming.pid"
    "${OUTPUT_ROOT}/${dataset}/rekv/rekv_topk${REKV_TOPK}_nlocal${REKV_N_LOCAL}.pid"
    "${OUTPUT_ROOT}/${dataset}/duo_streaming/duo_streaming_s075_sink${SINK}_recent${RECENT}.pid"
    "${OUTPUT_ROOT}/${dataset}/duo_plus_rekv/duo_plus_rekv_s075_sink${SINK}_recent${RECENT}_topk${REKV_TOPK}.pid"
  )
  local labels=(full_streaming rekv duo_streaming duo_plus_rekv)
  local failed=0
  for i in "${!pid_files[@]}"; do
    local pid
    pid=$(cat "${pid_files[$i]}")
    local rc=0
    wait "${pid}" || rc=$?
    if [[ ${rc} -eq 0 ]]; then
      echo "[gpu${i}] ${labels[$i]} (pid ${pid}) — OK"
    else
      echo "[gpu${i}] ${labels[$i]} (pid ${pid}) — FAILED (exit ${rc})" >&2
      failed=1
    fi
  done
  return ${failed}
}

post_process() {
  local dataset=$1
  echo "=== Post-processing: ${dataset} ==="
  local files=()
  for method_dir in full_streaming duo_streaming rekv duo_plus_rekv; do
    while IFS= read -r f; do
      files+=("$f")
    done < <(find "${OUTPUT_ROOT}/${dataset}/${method_dir}" -maxdepth 1 -name "*.json" 2>/dev/null | sort)
  done
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No result files found — skipping." >&2
    return
  fi

  echo "[judge] Running LLM judge (in-place)..."
  $PYTHON -m streaming.ReKV.judge_results "${files[@]}" --in-place
  echo "[judge] Done."

  $PYTHON -m streaming.ReKV.compare_subsamples "${files[@]}" \
    --output-dir "${OUTPUT_ROOT}/${dataset}/comparison"
  echo "[comparison] ${OUTPUT_ROOT}/${dataset}/comparison/summary.md"

  $PYTHON -m streaming.ReKV.plot_results "${files[@]}" \
    --output-dir "${OUTPUT_ROOT}/${dataset}/plots"
  echo "[plots] ${OUTPUT_ROOT}/${dataset}/plots/"
  echo ""
}

for DATASET in ${DATASETS}; do
  echo "====== Dataset: ${DATASET} ======"

  run_method_bg 0 "${DATASET}" full_streaming "full_streaming"
  run_method_bg 1 "${DATASET}" rekv \
      "rekv_topk${REKV_TOPK}_nlocal${REKV_N_LOCAL}" \
      --retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}"
  run_method_bg 2 "${DATASET}" duo_streaming \
      "duo_streaming_s075_sink${SINK}_recent${RECENT}" \
      --attn-dir "${ATTN_DIR}" --sparsity "${SPARSITY}" \
      --deploy-sink-size "${SINK}" --deploy-recent-size "${RECENT}"
  run_method_bg 3 "${DATASET}" duo_plus_rekv \
      "duo_plus_rekv_s075_sink${SINK}_recent${RECENT}_topk${REKV_TOPK}" \
      --attn-dir "${ATTN_DIR}" --sparsity "${SPARSITY}" \
      --deploy-sink-size "${SINK}" --deploy-recent-size "${RECENT}" \
      --retrieve-size "${REKV_TOPK}" --n-local "${REKV_N_LOCAL}"

  echo "Waiting for all 4 methods..."
  if ! wait_all "${DATASET}"; then
    echo "" >&2
    echo "One or more methods failed for ${DATASET}. Last 10 lines of each log:" >&2
    find "${OUTPUT_ROOT}/${DATASET}" -name "*.log" | sort | while read -r f; do
      echo "=== $f ===" >&2
      tail -10 "$f" >&2
      echo "" >&2
    done
    exit 1
  fi

  post_process "${DATASET}"
done

echo "====== Full eval complete ======"
echo "Results: ${OUTPUT_ROOT}/"
