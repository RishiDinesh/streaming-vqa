#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

cd "${ROOT}"

# ReKV-aligned base dataset/eval config
export DATASET=${DATASET:-rvs_ego}
export HF_REPO_ID=${HF_REPO_ID:-Becomebright/RVS}
export SAMPLE_FPS=${SAMPLE_FPS:-0.5}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
export VIDEO_DECODE_THREADS=${VIDEO_DECODE_THREADS:-1}

# Only changed for OOM mitigation: more chunks => smaller per-chunk workload
export NUM_CHUNKS=${NUM_CHUNKS:-20}

# Output target requested by user
export OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/outputs/evaluations_streaming/untracked/rvs-ego}

# Resume-safe behavior:
# - RESUME=1 => do NOT force overwrite
# - RESUME=0 => default to overwrite for fresh runs
export RESUME=${RESUME:-0}
if [[ -z "${EXTRA_ARGS:-}" ]]; then
  if [[ "${RESUME}" == "1" ]]; then
    export EXTRA_ARGS=""
  else
    export EXTRA_ARGS="--overwrite-output"
  fi
else
  export EXTRA_ARGS
fi

if [[ -n "${METHOD:-}" ]]; then
  ORCH_MODE=${ORCH_MODE:-submit}
else
  ORCH_MODE=${ORCH_MODE:-all}
fi

bash "${SCRIPT_DIR}/run_streamingtom_vs_duo_plus_run2.sh" "${ORCH_MODE}"
