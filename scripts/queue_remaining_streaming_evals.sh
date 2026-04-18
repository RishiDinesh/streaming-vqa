#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/queue_remaining_streaming_evals.sh

This helper waits for any active full-eval launcher to finish, then resumes the
standard 0.5B topk=64 full-eval lane for:
  1. RVS-Ego
  2. RVS-Movie

Environment overrides:
  MODEL_05B
  ATTN_DIR_05B
  AB_SPARSITY
  REKV_N_LOCAL
  AB_N_LOCAL
  AB_DEPLOY_SINK_SIZE
  AB_DEPLOY_RECENT_SIZE
  WAIT_POLL_SEC
EOF
}

if [[ $# -ne 0 ]]; then
  usage >&2
  exit 1
fi

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

MODEL_05B=${MODEL_05B:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
ATTN_DIR_05B=${ATTN_DIR_05B:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
AB_SPARSITY=${AB_SPARSITY:-0.75}
REKV_N_LOCAL=${REKV_N_LOCAL:-15000}
AB_N_LOCAL=${AB_N_LOCAL:-15000}
AB_DEPLOY_SINK_SIZE=${AB_DEPLOY_SINK_SIZE:-256}
AB_DEPLOY_RECENT_SIZE=${AB_DEPLOY_RECENT_SIZE:-512}
WAIT_POLL_SEC=${WAIT_POLL_SEC:-60}

wait_for_active_full_eval() {
  while pgrep -f "scripts/run_streaming_full_eval_local.sh all" >/dev/null 2>&1; do
    echo "[queue] waiting for active full eval to finish at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    sleep "${WAIT_POLL_SEC}"
  done
}

run_cmd() {
  echo "[queue] starting $(date -u +"%Y-%m-%dT%H:%M:%SZ"): $*"
  "$@"
  echo "[queue] finished $(date -u +"%Y-%m-%dT%H:%M:%SZ"): $*"
}

wait_for_active_full_eval

run_cmd env \
  DATASET=rvs_ego \
  MODEL="${MODEL_05B}" \
  USE_FEATURE_CACHE=1 \
  OUTPUT_ROOT=outputs/evaluations_streaming/rvs-ego/full_eval_topk64_memavg \
  REKV_TOPK=64 \
  AB_TOPK=64 \
  REKV_N_LOCAL="${REKV_N_LOCAL}" \
  AB_N_LOCAL="${AB_N_LOCAL}" \
  AB_SPARSITY="${AB_SPARSITY}" \
  AB_DEPLOY_SINK_SIZE="${AB_DEPLOY_SINK_SIZE}" \
  AB_DEPLOY_RECENT_SIZE="${AB_DEPLOY_RECENT_SIZE}" \
  ATTN_DIR="${ATTN_DIR_05B}" \
  RESUME=1 \
  bash scripts/run_streaming_full_eval_local.sh all

run_cmd env \
  DATASET=rvs_movie \
  MODEL="${MODEL_05B}" \
  USE_FEATURE_CACHE=1 \
  OUTPUT_ROOT=outputs/evaluations_streaming/rvs-movie/full_eval_topk64_memavg \
  REKV_TOPK=64 \
  AB_TOPK=64 \
  REKV_N_LOCAL="${REKV_N_LOCAL}" \
  AB_N_LOCAL="${AB_N_LOCAL}" \
  AB_SPARSITY="${AB_SPARSITY}" \
  AB_DEPLOY_SINK_SIZE="${AB_DEPLOY_SINK_SIZE}" \
  AB_DEPLOY_RECENT_SIZE="${AB_DEPLOY_RECENT_SIZE}" \
  ATTN_DIR="${ATTN_DIR_05B}" \
  RESUME=1 \
  bash scripts/run_streaming_full_eval_local.sh all
