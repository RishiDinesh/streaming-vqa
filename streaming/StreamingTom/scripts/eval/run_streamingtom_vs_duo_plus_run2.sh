#!/usr/bin/env bash
# Launch + postprocess StreamingTom vs Duo+StreamingTom in ReKV-style run2 layout.
# Output root (default): outputs/evaluations_streaming/untracked/<dataset>/full_eval/run2
#
# Modes:
#   submit     submit/launch chunk jobs for both methods
#   post       merge chunks + comparison + plots (+ optional judge)
#   all        submit then post (local mode only waits implicitly)
#
# Usage:
#   bash streaming/StreamingTom/scripts/eval/run_streamingtom_vs_duo_plus_run2.sh submit
#   bash streaming/StreamingTom/scripts/eval/run_streamingtom_vs_duo_plus_run2.sh post
#   bash streaming/StreamingTom/scripts/eval/run_streamingtom_vs_duo_plus_run2.sh all

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <submit|post|all>" >&2
  exit 1
fi

MODE=$1
ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)
cd "${ROOT}"

# StreamingTom requires the duo-st env (torch 2.5.1+cu124, flashinfer, LLaVA-NeXT).
# Do NOT use activate_streaming_env — it picks up envs/duo (the ReKV env).
_CONDA_INIT=""
for _c in \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "/root/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniforge3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh" \
  "/u/navdeep/miniconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${_c}" ]]; then _CONDA_INIT="${_c}"; break; fi
done
if [[ -z "${_CONDA_INIT}" ]]; then
  echo "[error] Cannot find conda init script" >&2; exit 1
fi
# shellcheck disable=SC1090
source "${_CONDA_INIT}"
DUO_ST_ENV="${ROOT}/envs/duo-st"
if [[ -d "${DUO_ST_ENV}" ]]; then
  conda activate "${DUO_ST_ENV}"
  echo "[env] Activated duo-st env: ${DUO_ST_ENV}" >&2
else
  echo "[error] duo-st env not found at ${DUO_ST_ENV}" >&2
  echo "[error] Run: bash streaming/StreamingTom/scripts/setup_duo_st_env.sh" >&2
  exit 1
fi
export PYTHONPATH="${ROOT}:${ROOT}/streaming/StreamingTom:${PYTHONPATH:-}"

DATASET=${DATASET:-rvs_ego}
NUM_CHUNKS=${NUM_CHUNKS:-10}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/outputs/evaluations_streaming/untracked/${DATASET//_/-}/full_eval/run2}
ARRAY_RANGE=${ARRAY_RANGE:-0-$((NUM_CHUNKS-1))}

if [[ -n "${METHOD:-}" ]]; then
  case "${METHOD}" in
    streamingtom|duo_plus_streamingtom)
      METHODS=("${METHOD}")
      ;;
    *)
      echo "[error] METHOD must be one of: streamingtom, duo_plus_streamingtom" >&2
      exit 1
      ;;
  esac
else
  METHODS=(streamingtom duo_plus_streamingtom)
fi

submit_local() {
  local method=$1
  local idx
  for ((idx=0; idx<NUM_CHUNKS; idx++)); do
    echo "[local] method=${method} chunk=${idx}/${NUM_CHUNKS}"
    SLURM_ARRAY_TASK_ID=${idx} \
    NUM_CHUNKS=${NUM_CHUNKS} \
    DATASET=${DATASET} \
    METHOD=${method} \
    OUTPUT_ROOT=${OUTPUT_ROOT} \
    bash streaming/StreamingTom/scripts/eval/run_streamingtom_eval_slurm_array.sh
  done
}

submit_slurm() {
  local method=$1
  sbatch --array="${ARRAY_RANGE}" \
    --output="${ROOT}/logs/streamingtom-${method}-%a-%j.out" \
    --export=ALL,DATASET="${DATASET}",NUM_CHUNKS="${NUM_CHUNKS}",METHOD="${method}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
    streaming/StreamingTom/scripts/eval/run_streamingtom_eval_slurm_array.sh
}

run_submit() {
  mkdir -p "${OUTPUT_ROOT}" "${ROOT}/logs"
  if command -v sbatch >/dev/null 2>&1; then
    echo "[submit] using sbatch arrays (${ARRAY_RANGE})"
    for method in "${METHODS[@]}"; do
      submit_slurm "${method}"
    done
    echo "[submit] submitted arrays for: ${METHODS[*]}"
  else
    echo "[submit] sbatch not found; running local chunk loop"
    for method in "${METHODS[@]}"; do
      submit_local "${method}"
    done
  fi
}

run_post() {
  mkdir -p \
    "${OUTPUT_ROOT}/streamingtom" \
    "${OUTPUT_ROOT}/duo_plus_streamingtom" \
    "${OUTPUT_ROOT}/merged" \
    "${OUTPUT_ROOT}/comparison" \
    "${OUTPUT_ROOT}/plots" \
    "${OUTPUT_ROOT}/plots_judge"

  python streaming/StreamingTom/scripts/eval/merge_chunks.py \
    --run-root "${OUTPUT_ROOT}" \
    --methods streamingtom duo_plus_streamingtom

  if [[ "${RUN_JUDGE:-0}" == "1" ]]; then
    python -m streaming.ReKV.judge_results \
      --in-place \
      "${OUTPUT_ROOT}/merged/streamingtom.json" \
      "${OUTPUT_ROOT}/merged/duo_plus_streamingtom.json"

    python -m streaming.ReKV.plot_results \
      "${OUTPUT_ROOT}/merged/streamingtom.json" \
      "${OUTPUT_ROOT}/merged/duo_plus_streamingtom.json" \
      --output-dir "${OUTPUT_ROOT}/plots_judge"
  fi

  echo "[post] artifacts ready under: ${OUTPUT_ROOT}"
}

case "${MODE}" in
  submit)
    run_submit
    ;;
  post)
    run_post
    ;;
  all)
    run_submit
    run_post
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac
