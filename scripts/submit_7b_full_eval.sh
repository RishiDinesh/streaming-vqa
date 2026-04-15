#!/usr/bin/env bash
# Submit full 7B eval for both datasets (rvs_movie + rvs_ego), all 4 methods.
# Uses NUM_CHUNKS=10 + --mem=120G for both datasets to avoid OOM.
#
# Flow:
#   1. Smoke test (array 0-3, 1 video each, rvs_movie) is submitted first.
#   2. Full eval jobs are submitted with --dependency=afterok:<smoke_jobid>
#      so they only run if ALL 4 smoke tasks succeed.
#
# Usage (from repo root):
#   bash scripts/submit_7b_full_eval.sh
#
# Outputs:
#   outputs/evaluations_streaming/rvs-movie/7b_full_eval/
#   outputs/evaluations_streaming/rvs-ego/7b_full_eval/

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
cd "${ROOT}"
mkdir -p logs

MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
ATTN_DIR="outputs/train/7b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5"
SPARSITY="0.75"
DEPLOY_SINK_SIZE="256"
DEPLOY_RECENT_SIZE="512"
NUM_CHUNKS=10

MOVIE_OUTPUT="${ROOT}/outputs/evaluations_streaming/rvs-movie/7b_full_eval"
EGO_OUTPUT="${ROOT}/outputs/evaluations_streaming/rvs-ego/7b_full_eval"

# ── Step 1: smoke test (already submitted as 151546, reuse if still pending) ─
echo "=== Step 1: Smoke test ==="
SMOKE_JOB=$(sbatch \
  --array=0-3 \
  --mem=120G \
  --output="logs/7b-smoke-%a-%j.out" \
  --parsable \
  scripts/run_7b_smoke_test.sh)
echo "Smoke test job: ${SMOKE_JOB}"

# ── Step 2: full eval — rvs-movie, 4 methods × 10 chunks ────────────────────
echo ""
echo "=== Step 2: Full eval — rvs-movie (dependency: afterok:${SMOKE_JOB}) ==="
for METHOD in full_streaming duo_streaming rekv duo_plus_rekv; do
  JID=$(DATASET=rvs_movie \
    MODEL="${MODEL}" \
    METHOD="${METHOD}" \
    NUM_CHUNKS=${NUM_CHUNKS} \
    SPARSITY="${SPARSITY}" \
    DEPLOY_SINK_SIZE="${DEPLOY_SINK_SIZE}" \
    DEPLOY_RECENT_SIZE="${DEPLOY_RECENT_SIZE}" \
    ATTN_DIR="${ATTN_DIR}" \
    OUTPUT_ROOT="${MOVIE_OUTPUT}" \
    sbatch \
      --array=0-$((NUM_CHUNKS - 1)) \
      --mem=120G \
      --output="logs/7b-movie-${METHOD}-%a-%j.out" \
      --dependency="afterok:${SMOKE_JOB}" \
      --parsable \
      scripts/run_streaming_eval_slurm_array.sh)
  echo "  rvs-movie/${METHOD}: job ${JID}"
done

# ── Step 3: full eval — rvs-ego, 4 methods × 10 chunks ─────────────────────
echo ""
echo "=== Step 3: Full eval — rvs-ego (dependency: afterok:${SMOKE_JOB}) ==="
for METHOD in full_streaming duo_streaming rekv duo_plus_rekv; do
  JID=$(DATASET=rvs_ego \
    MODEL="${MODEL}" \
    METHOD="${METHOD}" \
    NUM_CHUNKS=${NUM_CHUNKS} \
    SPARSITY="${SPARSITY}" \
    DEPLOY_SINK_SIZE="${DEPLOY_SINK_SIZE}" \
    DEPLOY_RECENT_SIZE="${DEPLOY_RECENT_SIZE}" \
    ATTN_DIR="${ATTN_DIR}" \
    OUTPUT_ROOT="${EGO_OUTPUT}" \
    sbatch \
      --array=0-$((NUM_CHUNKS - 1)) \
      --mem=120G \
      --output="logs/7b-ego-${METHOD}-%a-%j.out" \
      --dependency="afterok:${SMOKE_JOB}" \
      --parsable \
      scripts/run_streaming_eval_slurm_array.sh)
  echo "  rvs-ego/${METHOD}: job ${JID}"
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: watch -n 30 'squeue -u navdeep'"
echo "Outputs:"
echo "  ${MOVIE_OUTPUT}/"
echo "  ${EGO_OUTPUT}/"
