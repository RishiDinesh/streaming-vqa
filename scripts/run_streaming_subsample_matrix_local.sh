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
  scripts/run_streaming_subsample_matrix_local.sh <mode>

Modes:
  ego     Run both RVS-Ego subsample slices
  movie   Run both RVS-Movie subsample slices
  all     Run all four subsample slices
  status  Print the current status file

Environment overrides:
  RUN_ROOT
  STATUS_FILE
  LOG_ROOT
  MONITOR_INTERVAL_SEC
  FEATURE_CACHE_ROOT
  USE_FEATURE_CACHE
  MAX_VIDEOS
  MAX_CONVERSATIONS
  SAMPLE_FPS
  MAX_NEW_TOKENS
  VIDEO_DECODE_THREADS
  CLEAR_CUDA_CACHE_ON_RESET
  FLUSH_EVERY_CONVERSATIONS
  ATTN_DIR
  OUTPUT_SUFFIX
  REKV_TOPK
  REKV_N_LOCAL
  AB_TOPK
  AB_N_LOCAL
  AB_DEPLOY_SINK_SIZE
  AB_DEPLOY_RECENT_SIZE
  EXTRA_ARGS
EOF
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

MODE=$1
ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-outputs/evaluations_streaming/subsample_runner}
STATUS_FILE=${STATUS_FILE:-${RUN_ROOT}/status.txt}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
MONITOR_INTERVAL_SEC=${MONITOR_INTERVAL_SEC:-20}
MAX_VIDEOS=${MAX_VIDEOS:-5}
MAX_CONVERSATIONS=${MAX_CONVERSATIONS:-3}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
FEATURE_CACHE_ROOT=${FEATURE_CACHE_ROOT:-}
USE_FEATURE_CACHE=${USE_FEATURE_CACHE:-0}
VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-4}
CLEAR_CUDA_CACHE_ON_RESET=${CLEAR_CUDA_CACHE_ON_RESET:-0}
FLUSH_EVERY_CONVERSATIONS=${FLUSH_EVERY_CONVERSATIONS:-1}
ATTN_DIR=${ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-}
REKV_TOPK=${REKV_TOPK:-64}
REKV_N_LOCAL=${REKV_N_LOCAL:-15000}
AB_TOPK=${AB_TOPK:-${REKV_TOPK}}
AB_N_LOCAL=${AB_N_LOCAL:-${REKV_N_LOCAL}}
AB_DEPLOY_SINK_SIZE=${AB_DEPLOY_SINK_SIZE:-}
AB_DEPLOY_RECENT_SIZE=${AB_DEPLOY_RECENT_SIZE:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

activate_duo_env

cd "${ROOT}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

if [[ "${MODE}" == "status" ]]; then
  if [[ -f "${STATUS_FILE}" ]]; then
    cat "${STATUS_FILE}"
    exit 0
  fi
  echo "No status file found at ${STATUS_FILE}" >&2
  exit 1
fi

declare -a SLICES=()
case "${MODE}" in
  ego)
    SLICES+=("rvs_ego|subsample5|0")
    SLICES+=("rvs_ego|subsample5_offset5|5")
    ;;
  movie)
    SLICES+=("rvs_movie|subsample5_movie|0")
    SLICES+=("rvs_movie|subsample5_movie_offset5|5")
    ;;
  all)
    SLICES+=("rvs_ego|subsample5|0")
    SLICES+=("rvs_ego|subsample5_offset5|5")
    SLICES+=("rvs_movie|subsample5_movie|0")
    SLICES+=("rvs_movie|subsample5_movie_offset5|5")
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

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

slice_output_root() {
  local dataset_dash=$1
  local slice=$2
  local root="outputs/evaluations_streaming/${dataset_dash}/${slice}"
  if [[ -n "${OUTPUT_SUFFIX}" ]]; then
    root+="_${OUTPUT_SUFFIX}"
  fi
  printf '%s' "${root}"
}

declare -a STEPS=(
  "full|full_streaming/full_streaming.json"
  "duo|duo_streaming/duo_streaming_s05.json"
  "rekv|rekv/$(build_rekv_tag).json"
  "ab_s05|duo_plus_rekv/$(build_ab_tag 0.5).json"
  "ab_s075|duo_plus_rekv/$(build_ab_tag 0.75).json"
  "judge|"
  "plots|"
  "qualitative|"
)

TOTAL_STEPS=$(( ${#SLICES[@]} * ${#STEPS[@]} ))
CURRENT_STEP=0

write_status() {
  local body=$1
  cat > "${STATUS_FILE}" <<EOF
updated_at_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
${body}
EOF
}

render_output_path() {
  local relative_path=$1
  local dataset=$2
  local slice=$3
  local dataset_dash=${dataset//_/-}
  printf '%s/%s' "$(slice_output_root "${dataset_dash}" "${slice}")" "${relative_path}"
}

monitor_checkpoint() {
  local label=$1
  local output_path=$2
  local log_path=$3
  local step_index=$4
  while true; do
    if [[ ! -f "${output_path}" ]]; then
      write_status "phase: running
step: ${step_index}/${TOTAL_STEPS}
label: ${label}
log: ${log_path}
checkpoint: pending"
      sleep "${MONITOR_INTERVAL_SEC}"
      continue
    fi

    python - "${output_path}" "${STATUS_FILE}" "${label}" "${log_path}" "${step_index}" "${TOTAL_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])
label = sys.argv[3]
log_path = sys.argv[4]
step_index = sys.argv[5]
total_steps = sys.argv[6]

with open(output_path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

run_state = payload.get("run_state", {})
in_progress = payload.get("in_progress_video") or {}
checkpoint_state = in_progress.get("checkpoint_state") or {}

lines = [
    f"updated_at_utc: __DATE__",
    "phase: running",
    f"step: {step_index}/{total_steps}",
    f"label: {label}",
    f"log: {log_path}",
    f"checkpoint: {output_path}",
    f"run_status: {run_state.get('status')}",
    f"completed_videos: {run_state.get('completed_videos')}/{run_state.get('total_requested_videos')}",
    f"in_progress_video: {in_progress.get('video_id')}",
    f"in_progress_conversations: {checkpoint_state.get('completed_conversations')}",
    f"in_progress_frames: {checkpoint_state.get('frames_ingested')}",
    f"json_updated_at_utc: {run_state.get('updated_at_utc')}",
]
status_path.write_text("\n".join(lines).replace("__DATE__", __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")) + "\n", encoding="utf-8")
PY

    sleep "${MONITOR_INTERVAL_SEC}"
  done
}

run_step() {
  local dataset=$1
  local slice=$2
  local offset=$3
  local mode=$4
  local output_template=$5
  local dataset_dash=${dataset//_/-}
  local label="${dataset}:${slice}:${mode}"
  local log_path="${LOG_ROOT}/${dataset_dash}_${slice}_${mode}.log"
  local output_path=""
  local monitor_pid=""
  local current_output_root

  CURRENT_STEP=$((CURRENT_STEP + 1))

  if [[ -n "${output_template}" ]]; then
    output_path=$(render_output_path "${output_template}" "${dataset}" "${slice}")
  fi
  current_output_root=$(slice_output_root "${dataset_dash}" "${slice}")

  if [[ "${mode}" == "plots" ]]; then
    rm -rf "${current_output_root}/plots"
  elif [[ "${mode}" == "qualitative" ]]; then
    rm -rf "${current_output_root}/qualitative"
  fi

  write_status "phase: launching
step: ${CURRENT_STEP}/${TOTAL_STEPS}
label: ${label}
log: ${log_path}
checkpoint: ${output_path:-none}"

  echo
  echo "[matrix ${CURRENT_STEP}/${TOTAL_STEPS}] ${label}"
  echo "log: ${log_path}"
  if [[ -n "${output_path}" ]]; then
    echo "checkpoint: ${output_path}"
  fi

  (
    set -o pipefail
    env \
      DATASET="${dataset}" \
      SUBSAMPLE_NAME="${slice}" \
      VIDEO_OFFSET="${offset}" \
      MAX_VIDEOS="${MAX_VIDEOS}" \
      MAX_CONVERSATIONS="${MAX_CONVERSATIONS}" \
      SAMPLE_FPS="${SAMPLE_FPS}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT}" \
      USE_FEATURE_CACHE="${USE_FEATURE_CACHE}" \
      VIDEO_DECODE_THREADS="${VIDEO_DECODE_THREADS}" \
      CLEAR_CUDA_CACHE_ON_RESET="${CLEAR_CUDA_CACHE_ON_RESET}" \
      FLUSH_EVERY_CONVERSATIONS="${FLUSH_EVERY_CONVERSATIONS}" \
      ATTN_DIR="${ATTN_DIR}" \
      OUTPUT_ROOT="${current_output_root}" \
      REKV_TOPK="${REKV_TOPK}" \
      REKV_N_LOCAL="${REKV_N_LOCAL}" \
      AB_TOPK="${AB_TOPK}" \
      AB_N_LOCAL="${AB_N_LOCAL}" \
      AB_DEPLOY_SINK_SIZE="${AB_DEPLOY_SINK_SIZE}" \
      AB_DEPLOY_RECENT_SIZE="${AB_DEPLOY_RECENT_SIZE}" \
      bash scripts/run_streaming_subsample5_local.sh "${mode}" ${EXTRA_ARGS} \
      2>&1 | tee "${log_path}"
  ) &
  local run_pid=$!

  if [[ -n "${output_path}" ]]; then
    monitor_checkpoint "${label}" "${output_path}" "${log_path}" "${CURRENT_STEP}" &
    monitor_pid=$!
  fi

  local rc=0
  set +e
  wait "${run_pid}"
  rc=$?
  set -e

  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi

  if [[ ${rc} -ne 0 ]]; then
    write_status "phase: failed
step: ${CURRENT_STEP}/${TOTAL_STEPS}
label: ${label}
log: ${log_path}
checkpoint: ${output_path:-none}
exit_code: ${rc}"
    echo "[matrix] failed: ${label}" >&2
    exit "${rc}"
  fi

  write_status "phase: completed_step
step: ${CURRENT_STEP}/${TOTAL_STEPS}
label: ${label}
log: ${log_path}
checkpoint: ${output_path:-none}
exit_code: 0"
}

run_compare_bundle() {
  local dataset=$1
  local slice_a=$2
  local slice_b=$3
  local output_dir=$4
  local dataset_dash=${dataset//_/-}
  local root_a
  local root_b
  local files=()
  root_a=$(slice_output_root "${dataset_dash}" "${slice_a}")
  root_b=$(slice_output_root "${dataset_dash}" "${slice_b}")
  local root
  local method_dir
  for root in "${root_a}" "${root_b}"; do
    for method_dir in full_streaming duo_streaming rekv duo_plus_rekv; do
      if [[ -d "${root}/${method_dir}" ]]; then
        while IFS= read -r path; do
          files+=("${path}")
        done < <(find "${root}/${method_dir}" -maxdepth 1 -type f -name '*.json' | sort)
      fi
    done
  done
  python -m streaming.ReKV.compare_subsamples "${files[@]}" --output-dir "${output_dir}"
}

comparison_output_dir() {
  local base_dir=$1
  if [[ -n "${OUTPUT_SUFFIX}" ]]; then
    printf '%s_%s' "${base_dir}" "${OUTPUT_SUFFIX}"
  else
    printf '%s' "${base_dir}"
  fi
}

for entry in "${SLICES[@]}"; do
  IFS='|' read -r dataset slice offset <<< "${entry}"
  for step in "${STEPS[@]}"; do
    IFS='|' read -r mode output_template <<< "${step}"
    run_step "${dataset}" "${slice}" "${offset}" "${mode}" "${output_template}"
  done
done

if [[ "${MODE}" == "ego" || "${MODE}" == "all" ]]; then
  run_compare_bundle \
    "rvs_ego" \
    "subsample5" \
    "subsample5_offset5" \
    "$(comparison_output_dir "outputs/evaluations_streaming/rvs-ego/subsample_comparison_offset0_vs_offset5")"
fi

if [[ "${MODE}" == "movie" || "${MODE}" == "all" ]]; then
  run_compare_bundle \
    "rvs_movie" \
    "subsample5_movie" \
    "subsample5_movie_offset5" \
    "$(comparison_output_dir "outputs/evaluations_streaming/rvs-movie/subsample_comparison_offset0_vs_offset5")"
fi

write_status "phase: complete
step: ${CURRENT_STEP}/${TOTAL_STEPS}
label: all_done
log_root: ${LOG_ROOT}"

echo
echo "[matrix] all requested subsample steps completed"
echo "status: ${STATUS_FILE}"
echo "logs:   ${LOG_ROOT}"
