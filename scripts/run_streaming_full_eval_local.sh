#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_streaming_full_eval_local.sh <mode>

Modes:
  precompute      Build the shared visual feature cache only
  full            Run full_streaming
  duo             Run duo_streaming with sparsity=0.5
  duo_fullheads   Run duo_streaming with sparsity=0.0
  rekv            Run rekv with retrieve_size=64 and n_local=15000
  ab              Run duo_plus_rekv with sparsity=0.375 and ReKV defaults
  judge           Judge any existing promoted full-eval JSONs in place
  all             Run the official four methods from cache, then judge them in place
  all_with_control
                  Run the official four methods plus duo_fullheads, then judge them in place

Environment overrides:
  DATASET
  ANNOTATION_PATH
  VIDEO_ROOT
  HF_REPO_ID
  ALLOW_HF_VIDEO_DOWNLOAD
  MODEL
  OUTPUT_ROOT
  SAMPLE_FPS
  MAX_NEW_TOKENS
  MAX_VIDEOS
  MAX_CONVERSATIONS
  VIDEO_OFFSET
  FLUSH_EVERY_VIDEOS
  ATTN_DIR
  FEATURE_CACHE_ROOT
  FEATURE_BATCH_SIZE
  USE_FEATURE_CACHE
  RESUME
  OVERWRITE
  NUM_PROCESSES
  GPU_IDS
  PYTHON_BIN
  EXTRA_ARGS

Examples:
  DATASET=rvs_ego scripts/run_streaming_full_eval_local.sh precompute
  DATASET=rvs_ego scripts/run_streaming_full_eval_local.sh all
  DATASET=rvs_movie RESUME=1 scripts/run_streaming_full_eval_local.sh ab
  DATASET=rvs_movie NUM_PROCESSES=4 GPU_IDS=0,1,2,3 scripts/run_streaming_full_eval_local.sh all
  DATASET=rvs_movie scripts/run_streaming_full_eval_local.sh judge
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

MODE=$1
ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HELPER_PY="${ROOT}/scripts/streaming_parallel.py"

DATASET=${DATASET:-rvs_ego}
ANNOTATION_PATH=${ANNOTATION_PATH:-}
VIDEO_ROOT=${VIDEO_ROOT:-}
HF_REPO_ID=${HF_REPO_ID:-Becomebright/RVS}
ALLOW_HF_VIDEO_DOWNLOAD=${ALLOW_HF_VIDEO_DOWNLOAD:-1}
MODEL=${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
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
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
RESUME=${RESUME:-1}
OVERWRITE=${OVERWRITE:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}
EXTRA_ARGS=${EXTRA_ARGS:-}

source /root/miniforge3/etc/profile.d/conda.sh
conda activate duo

cd "${ROOT}"
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}

BASE_ARGS=(
  --dataset "${DATASET}"
  --hf-repo-id "${HF_REPO_ID}"
  --model "${MODEL}"
  --sample-fps "${SAMPLE_FPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --flush-every-videos "${FLUSH_EVERY_VIDEOS}"
)

if [[ -n "${ANNOTATION_PATH}" ]]; then
  BASE_ARGS+=(--annotation-path "${ANNOTATION_PATH}")
fi
if [[ -n "${VIDEO_ROOT}" ]]; then
  BASE_ARGS+=(--video-root "${VIDEO_ROOT}")
fi
if [[ "${ALLOW_HF_VIDEO_DOWNLOAD}" != "0" ]]; then
  BASE_ARGS+=(--allow-hf-video-download)
fi

PARALLEL_GPU_IDS=()

append_video_slice_args() {
  local -n target=$1
  local slice_offset=$2
  local slice_max_videos=$3
  target+=(--video-offset "${slice_offset}")
  if [[ -n "${slice_max_videos}" ]]; then
    target+=(--max-videos "${slice_max_videos}")
  fi
}

count_requested_videos() {
  local args=(
    count-samples
    --dataset "${DATASET}"
    --hf-repo-id "${HF_REPO_ID}"
    --video-offset "${VIDEO_OFFSET}"
  )
  if [[ -n "${ANNOTATION_PATH}" ]]; then
    args+=(--annotation-path "${ANNOTATION_PATH}")
  fi
  if [[ -n "${VIDEO_ROOT}" ]]; then
    args+=(--video-root "${VIDEO_ROOT}")
  fi
  if [[ "${ALLOW_HF_VIDEO_DOWNLOAD}" != "0" ]]; then
    args+=(--allow-hf-video-download)
  fi
  if [[ -n "${MAX_VIDEOS}" ]]; then
    args+=(--max-videos "${MAX_VIDEOS}")
  fi
  "${PYTHON_BIN}" "${HELPER_PY}" "${args[@]}"
}

parse_gpu_ids() {
  local gpu_ids_clean=${GPU_IDS// /}
  IFS=',' read -r -a PARALLEL_GPU_IDS <<< "${gpu_ids_clean}"
  if [[ ${#PARALLEL_GPU_IDS[@]} -eq 0 || -z "${PARALLEL_GPU_IDS[0]}" ]]; then
    echo "GPU_IDS must contain at least one GPU id." >&2
    exit 1
  fi
}

validate_parallel_configuration() {
  if (( NUM_PROCESSES < 1 )); then
    echo "NUM_PROCESSES must be >= 1." >&2
    exit 1
  fi
  if (( NUM_PROCESSES == 1 )); then
    return 0
  fi
  local forbidden_flags=(
    --device
    --video-offset
    --max-videos
    --video-id
    --video-index
    --output-path
    --resume
    --overwrite-output
    --feature-cache-root
    --feature-batch-size
    --overwrite-existing
  )
  local flag
  for flag in "${forbidden_flags[@]}"; do
    if [[ " ${EXTRA_ARGS} " == *" ${flag} "* || "${EXTRA_ARGS}" == *"${flag}="* ]]; then
      echo "EXTRA_ARGS may not contain ${flag} when NUM_PROCESSES>1." >&2
      exit 1
    fi
  done
  parse_gpu_ids
  if (( ${#PARALLEL_GPU_IDS[@]} < NUM_PROCESSES )); then
    echo "Need at least ${NUM_PROCESSES} GPU ids in GPU_IDS, found ${#PARALLEL_GPU_IDS[@]}." >&2
    exit 1
  fi
}

wait_for_workers() {
  local pid
  local status=0
  for pid in "$@"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

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

should_skip_final_output() {
  local output_path=$1
  if [[ ! -f "${output_path}" ]]; then
    return 1
  fi
  if [[ "${RESUME}" == "1" ]]; then
    echo "Final output already exists at ${output_path}; skipping."
    return 0
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    return 1
  fi
  echo "Output already exists: ${output_path}. Set RESUME=1 or OVERWRITE=1." >&2
  exit 1
}

run_single_eval() {
  local method=$1
  local tag=$2
  shift 2
  local output_path="${OUTPUT_ROOT}/${method}/${tag}.json"
  local args=("${BASE_ARGS[@]}")
  append_video_slice_args args "${VIDEO_OFFSET}" "${MAX_VIDEOS}"
  if [[ -f "${output_path}" && "${RESUME}" == "1" ]]; then
    args+=(--resume)
  elif [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite-output)
  fi
  if [[ "${USE_FEATURE_CACHE}" == "1" ]]; then
    args+=(--feature-cache-root "${FEATURE_CACHE_ROOT}")
  fi
  if [[ -n "${MAX_CONVERSATIONS}" ]]; then
    args+=(--max-conversations-per-video "${MAX_CONVERSATIONS}")
  fi
  "${PYTHON_BIN}" -m streaming.ReKV.run_eval \
    "${args[@]}" \
    --method "${method}" \
    --output-path "${output_path}" \
    "$@" \
    ${EXTRA_ARGS}
}

run_parallel_eval() {
  local method=$1
  local tag=$2
  shift 2
  local method_args=("$@")
  local output_path="${OUTPUT_ROOT}/${method}/${tag}.json"
  if should_skip_final_output "${output_path}"; then
    return 0
  fi

  local total_videos
  total_videos=$(count_requested_videos)
  if (( total_videos <= 0 )); then
    echo "No videos matched the requested dataset filters." >&2
    exit 1
  fi

  local worker_count=${NUM_PROCESSES}
  if (( worker_count > total_videos )); then
    worker_count=${total_videos}
  fi
  if (( worker_count <= 1 )); then
    run_single_eval "${method}" "${tag}" "${method_args[@]}"
    return 0
  fi

  local shard_dir="${output_path%.json}.shards"
  mkdir -p "${shard_dir}"
  local base_chunk=$(( total_videos / worker_count ))
  local remainder=$(( total_videos % worker_count ))
  local local_offset=0
  local shard_size
  local shard_offset
  local shard_output
  local shard_gpu
  local args
  local -a shard_paths=()
  local -a pids=()
  local i

  for ((i = 0; i < worker_count; i++)); do
    shard_size=${base_chunk}
    if (( i < remainder )); then
      shard_size=$(( shard_size + 1 ))
    fi
    shard_offset=$(( VIDEO_OFFSET + local_offset ))
    shard_output="${shard_dir}/shard_${i}.json"
    shard_gpu=${PARALLEL_GPU_IDS[$i]}
    shard_paths+=("${shard_output}")
    args=("${BASE_ARGS[@]}")
    append_video_slice_args args "${shard_offset}" "${shard_size}"
    if [[ -f "${shard_output}" && "${RESUME}" == "1" ]]; then
      args+=(--resume)
    elif [[ "${OVERWRITE}" == "1" ]]; then
      args+=(--overwrite-output)
    fi
    if [[ "${USE_FEATURE_CACHE}" == "1" ]]; then
      args+=(--feature-cache-root "${FEATURE_CACHE_ROOT}")
    fi
    if [[ -n "${MAX_CONVERSATIONS}" ]]; then
      args+=(--max-conversations-per-video "${MAX_CONVERSATIONS}")
    fi
    # Each shard keeps the standard single-GPU run_eval path so latency and
    # memory metrics are measured exactly as they are in the non-sharded run.
    echo "[launch] method=${method} shard=$((i + 1))/${worker_count} gpu=${shard_gpu} offset=${shard_offset} max_videos=${shard_size}"
    CUDA_VISIBLE_DEVICES="${shard_gpu}" "${PYTHON_BIN}" -m streaming.ReKV.run_eval \
      "${args[@]}" \
      --method "${method}" \
      --output-path "${shard_output}" \
      "${method_args[@]}" \
      ${EXTRA_ARGS} &
    pids+=("$!")
    local_offset=$(( local_offset + shard_size ))
  done

  wait_for_workers "${pids[@]}"

  args=(
    merge-results
    --output-path "${output_path}"
    --video-offset "${VIDEO_OFFSET}"
    --total-requested-videos "${total_videos}"
  )
  if [[ -n "${MAX_VIDEOS}" ]]; then
    args+=(--max-videos "${MAX_VIDEOS}")
  fi
  for shard_output in "${shard_paths[@]}"; do
    args+=(--shard-path "${shard_output}")
  done
  "${PYTHON_BIN}" "${HELPER_PY}" "${args[@]}"
}

run_one() {
  local method=$1
  local tag=$2
  shift 2
  if (( NUM_PROCESSES > 1 )); then
    run_parallel_eval "${method}" "${tag}" "$@"
  else
    run_single_eval "${method}" "${tag}" "$@"
  fi
}

run_single_precompute() {
  local args=(
    --dataset "${DATASET}"
    --hf-repo-id "${HF_REPO_ID}"
    --model "${MODEL}"
    --sample-fps "${SAMPLE_FPS}"
    --feature-batch-size "${FEATURE_BATCH_SIZE}"
    --feature-cache-root "${FEATURE_CACHE_ROOT}"
  )
  if [[ -n "${ANNOTATION_PATH}" ]]; then
    args+=(--annotation-path "${ANNOTATION_PATH}")
  fi
  if [[ -n "${VIDEO_ROOT}" ]]; then
    args+=(--video-root "${VIDEO_ROOT}")
  fi
  if [[ "${ALLOW_HF_VIDEO_DOWNLOAD}" != "0" ]]; then
    args+=(--allow-hf-video-download)
  fi
  append_video_slice_args args "${VIDEO_OFFSET}" "${MAX_VIDEOS}"
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite-existing)
  fi
  "${PYTHON_BIN}" -m streaming.ReKV.precompute_features \
    "${args[@]}" \
    ${EXTRA_ARGS}
}

run_parallel_precompute() {
  local total_videos
  total_videos=$(count_requested_videos)
  if (( total_videos <= 0 )); then
    echo "No videos matched the requested dataset filters." >&2
    exit 1
  fi

  local worker_count=${NUM_PROCESSES}
  if (( worker_count > total_videos )); then
    worker_count=${total_videos}
  fi
  if (( worker_count <= 1 )); then
    run_single_precompute
    return 0
  fi

  local base_chunk=$(( total_videos / worker_count ))
  local remainder=$(( total_videos % worker_count ))
  local local_offset=0
  local shard_size
  local shard_offset
  local shard_gpu
  local args
  local -a pids=()
  local i

  for ((i = 0; i < worker_count; i++)); do
    shard_size=${base_chunk}
    if (( i < remainder )); then
      shard_size=$(( shard_size + 1 ))
    fi
    shard_offset=$(( VIDEO_OFFSET + local_offset ))
    shard_gpu=${PARALLEL_GPU_IDS[$i]}
    args=(
      --dataset "${DATASET}"
      --hf-repo-id "${HF_REPO_ID}"
      --model "${MODEL}"
      --sample-fps "${SAMPLE_FPS}"
      --feature-batch-size "${FEATURE_BATCH_SIZE}"
      --feature-cache-root "${FEATURE_CACHE_ROOT}"
    )
    if [[ -n "${ANNOTATION_PATH}" ]]; then
      args+=(--annotation-path "${ANNOTATION_PATH}")
    fi
    if [[ -n "${VIDEO_ROOT}" ]]; then
      args+=(--video-root "${VIDEO_ROOT}")
    fi
    if [[ "${ALLOW_HF_VIDEO_DOWNLOAD}" != "0" ]]; then
      args+=(--allow-hf-video-download)
    fi
    append_video_slice_args args "${shard_offset}" "${shard_size}"
    if [[ "${OVERWRITE}" == "1" ]]; then
      args+=(--overwrite-existing)
    fi
    echo "[launch] precompute shard=$((i + 1))/${worker_count} gpu=${shard_gpu} offset=${shard_offset} max_videos=${shard_size}"
    CUDA_VISIBLE_DEVICES="${shard_gpu}" "${PYTHON_BIN}" -m streaming.ReKV.precompute_features \
      "${args[@]}" \
      ${EXTRA_ARGS} &
    pids+=("$!")
    local_offset=$(( local_offset + shard_size ))
  done

  wait_for_workers "${pids[@]}"

  args=(
    write-cache-manifest
    --dataset "${DATASET}"
    --hf-repo-id "${HF_REPO_ID}"
    --model "${MODEL}"
    --sample-fps "${SAMPLE_FPS}"
    --feature-batch-size "${FEATURE_BATCH_SIZE}"
    --feature-cache-root "${FEATURE_CACHE_ROOT}"
    --video-offset "${VIDEO_OFFSET}"
  )
  if [[ -n "${ANNOTATION_PATH}" ]]; then
    args+=(--annotation-path "${ANNOTATION_PATH}")
  fi
  if [[ -n "${VIDEO_ROOT}" ]]; then
    args+=(--video-root "${VIDEO_ROOT}")
  fi
  if [[ "${ALLOW_HF_VIDEO_DOWNLOAD}" != "0" ]]; then
    args+=(--allow-hf-video-download)
  fi
  if [[ -n "${MAX_VIDEOS}" ]]; then
    args+=(--max-videos "${MAX_VIDEOS}")
  fi
  "${PYTHON_BIN}" "${HELPER_PY}" "${args[@]}"
}

run_precompute() {
  if (( NUM_PROCESSES > 1 )); then
    run_parallel_precompute
  else
    run_single_precompute
  fi
}

judge_all() {
  local files=()
  local maybe_files=(
    "${OUTPUT_ROOT}/full_streaming/full_streaming.json"
    "${OUTPUT_ROOT}/duo_streaming/duo_streaming_s05.json"
    "${OUTPUT_ROOT}/rekv/rekv_topk64_nlocal15000.json"
    "${OUTPUT_ROOT}/duo_plus_rekv/duo_plus_rekv_s0375_topk64_nlocal15000.json"
  )
  if [[ -f "${OUTPUT_ROOT}/duo_streaming/duo_streaming_s00.json" ]]; then
    maybe_files+=("${OUTPUT_ROOT}/duo_streaming/duo_streaming_s00.json")
  fi
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
  "${PYTHON_BIN}" -m streaming.ReKV.judge_results "${files[@]}" --in-place
}

if (( NUM_PROCESSES > 1 )) && [[ "${MODE}" != "judge" ]]; then
  validate_parallel_configuration
fi

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
  duo_fullheads)
    ensure_feature_cache_requested
    run_one duo_streaming duo_streaming_s00 --attn-dir "${ATTN_DIR}" --sparsity 0.0
    ;;
  rekv)
    ensure_feature_cache_requested
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    ;;
  ab)
    ensure_feature_cache_requested
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
    ensure_feature_cache_requested
    run_one full_streaming full_streaming
    run_one duo_streaming duo_streaming_s05 --attn-dir "${ATTN_DIR}" --sparsity 0.5
    run_one rekv rekv_topk64_nlocal15000 --retrieve-size 64 --n-local 15000
    run_one duo_plus_rekv duo_plus_rekv_s0375_topk64_nlocal15000 \
      --attn-dir "${ATTN_DIR}" \
      --sparsity 0.375 \
      --retrieve-size 64 \
      --n-local 15000
    judge_all
    ;;
  all_with_control)
    ensure_feature_cache_requested
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
