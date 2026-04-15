#!/usr/bin/env bash
# RunPod / bare-GPU parallel eval for StreamingTom methods 5-6.
# Runs both methods (streamingtom + duo_plus_streamingtom) in parallel across
# N GPUs using background processes — no SLURM required.
#
# Usage (from repo root, inside the duo-st conda env):
#
#   # 4 GPUs, rvs_ego, both methods:
#   NUM_GPUS=4 DATASET=rvs_ego \
#   bash streaming/StreamingTom/scripts/eval/run_eval_runpod.sh
#
#   # 2 GPUs, single method:
#   NUM_GPUS=2 DATASET=rvs_ego METHOD=streamingtom \
#   bash streaming/StreamingTom/scripts/eval/run_eval_runpod.sh
#
#   # Dry-run (print commands, don't execute):
#   DRY_RUN=1 NUM_GPUS=4 DATASET=rvs_ego \
#   bash streaming/StreamingTom/scripts/eval/run_eval_runpod.sh
#
# Prerequisites:
#   conda activate <repo>/envs/duo-st   (or: conda activate duo-st)
#   export PYTHONPATH=<repo>
#
# Each GPU handles one chunk. With NUM_GPUS=4 and NUM_CHUNKS=4 per method,
# each GPU processes ~2-3 videos.  Increase NUM_CHUNKS for finer sharding.
#
# Logs go to: <OUTPUT_ROOT>/logs/chunk_<i>_<method>.log

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
cd "${ROOT}"

export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# ── config ────────────────────────────────────────────────────────────────────
DATASET=${DATASET:-rvs_ego}
MODEL=${MODEL:-lmms-lab/llava-onevision-qwen2-0.5b-ov}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
NUM_GPUS=${NUM_GPUS:-4}
# Total chunks per method — set equal to NUM_GPUS for 1 chunk/GPU
NUM_CHUNKS=${NUM_CHUNKS:-${NUM_GPUS}}
RESUME=${RESUME:-1}
DRY_RUN=${DRY_RUN:-0}

# Which methods to run: space-separated list
METHODS=${METHODS:-"streamingtom duo_plus_streamingtom"}

OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/outputs/evaluations_streaming/${DATASET}/full_eval/run2}
STREAMINGTOM_ROOT=${STREAMINGTOM_ROOT:-streaming/StreamingTom}

DUO_ATTN_DIR=${DUO_ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
DUO_HEADS_FILE=${DUO_HEADS_FILE:-${DUO_ATTN_DIR}/full_attention_heads_latest.tsv}
DUO_THRESHOLD=${DUO_THRESHOLD:-0.5}
DUO_SPARSITY=${DUO_SPARSITY:-0.75}
DUO_SINK_SIZE=${DUO_SINK_SIZE:-256}
DUO_RECENT_SIZE=${DUO_RECENT_SIZE:-512}

# ── validate env ──────────────────────────────────────────────────────────────
echo "[runpod] Root:    ${ROOT}"
echo "[runpod] Dataset: ${DATASET}  Methods: ${METHODS}"
echo "[runpod] GPUs:    ${NUM_GPUS}  Chunks/method: ${NUM_CHUNKS}"
echo "[runpod] Output:  ${OUTPUT_ROOT}"
echo "[runpod] Python:  $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ "${NUM_CHUNKS}" -lt "${NUM_GPUS}" ]]; then
  echo "[warn] NUM_CHUNKS (${NUM_CHUNKS}) < NUM_GPUS (${NUM_GPUS}) — some GPUs will be idle"
fi

# ── launch chunks ─────────────────────────────────────────────────────────────
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PIDS=()
GPU_IDX=0

for METHOD in ${METHODS}; do
  for (( CHUNK=0; CHUNK<NUM_CHUNKS; CHUNK++ )); do
    GPU=$(( GPU_IDX % NUM_GPUS ))
    OUTPUT_PATH="${OUTPUT_ROOT}/${METHOD}/chunk_$(printf '%03d' "${CHUNK}").json"
    LOG_PATH="${LOG_DIR}/chunk_$(printf '%03d' "${CHUNK}")_${METHOD}.log"
    mkdir -p "$(dirname "${OUTPUT_PATH}")"

    CMD=(
      python streaming/StreamingTom/scripts/eval/run_eval.py
      --dataset "${DATASET}"
      --hf-repo-id Becomebright/RVS
      --allow-hf-video-download
      --model "${MODEL}"
      --method "${METHOD}"
      --sample-fps "${SAMPLE_FPS}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --video-decode-threads 1
      --num-chunks "${NUM_CHUNKS}"
      --chunk-index "${CHUNK}"
      --output-path "${OUTPUT_PATH}"
      --streamingtom-root "${STREAMINGTOM_ROOT}"
    )

    if [[ "${RESUME}" == "1" ]]; then
      CMD+=(--resume)
    fi

    if [[ "${METHOD}" == "duo_plus_streamingtom" ]]; then
      CMD+=(
        --duo-attn-dir "${DUO_ATTN_DIR}"
        --duo-heads-file "${DUO_HEADS_FILE}"
        --duo-threshold "${DUO_THRESHOLD}"
        --duo-sparsity "${DUO_SPARSITY}"
        --duo-sink-size "${DUO_SINK_SIZE}"
        --duo-recent-size "${DUO_RECENT_SIZE}"
      )
    fi

    echo "[runpod] GPU${GPU} chunk=${CHUNK}/${NUM_CHUNKS} method=${METHOD} → ${LOG_PATH}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "  DRY_RUN: CUDA_VISIBLE_DEVICES=${GPU} ${CMD[*]}"
    else
      CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
      PIDS+=($!)
    fi

    GPU_IDX=$(( GPU_IDX + 1 ))
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[runpod] DRY_RUN complete — no processes launched"
  exit 0
fi

# ── wait for all chunks ───────────────────────────────────────────────────────
echo ""
echo "[runpod] Launched ${#PIDS[@]} background workers. Waiting..."
FAIL=0
for PID in "${PIDS[@]}"; do
  if wait "${PID}"; then
    echo "[runpod] PID ${PID} done OK"
  else
    echo "[runpod] PID ${PID} FAILED (rc=$?)" >&2
    FAIL=$(( FAIL + 1 ))
  fi
done

echo ""
if [[ "${FAIL}" -gt 0 ]]; then
  echo "[runpod] ${FAIL} chunk(s) FAILED — check logs under ${LOG_DIR}" >&2
  exit 1
fi
echo "[runpod] All chunks complete."

# ── merge chunks ──────────────────────────────────────────────────────────────
echo ""
echo "[runpod] Merging chunks..."
python streaming/StreamingTom/scripts/eval/merge_chunks.py \
  --run-root "${OUTPUT_ROOT}" \
  --methods ${METHODS}

echo "[runpod] Done. Results under: ${OUTPUT_ROOT}"
echo "  Merged JSONs:  ${OUTPUT_ROOT}/merged/"
echo "  Comparison:    ${OUTPUT_ROOT}/comparison/"
echo "  Plots:         ${OUTPUT_ROOT}/plots/"
