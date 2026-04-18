#!/usr/bin/env bash
# Parallel eval for StreamingTom methods 5-6 across multiple datasets.
# Runs both methods × both datasets sequentially (one model loaded at a time),
# with chunks batched across GPUs (NUM_GPUS chunks in parallel per batch).
#
# Usage (from repo root):
#
#   # 2 GPUs, both methods, both datasets, 40 chunks each:
#   NUM_GPUS=2 NUM_CHUNKS=40 \
#   bash streaming/StreamingTom/scripts/eval/run_eval_runpod.sh
#
#   # Single method, single dataset override:
#   NUM_GPUS=2 NUM_CHUNKS=40 DATASETS="rvs_ego" METHODS="streamingtom" \
#   bash streaming/StreamingTom/scripts/eval/run_eval_runpod.sh
#
#   # Dry-run (print commands, don't execute):
#   DRY_RUN=1 NUM_GPUS=2 NUM_CHUNKS=4 \
#   bash streaming/StreamingTom/scripts/eval/run_eval_runpod.sh
#
# Chunk batching: only NUM_GPUS chunks run at a time — no OOM from simultaneous loads.
# Logs go to: <OUTPUT_ROOT>/<dataset>/<method>/logs/chunk_<i>.log

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
cd "${ROOT}"

# ── activate envs/duo-st ──────────────────────────────────────────────────────
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
  echo "[runpod] Activated: ${DUO_ST_ENV}" >&2
else
  echo "[error] duo-st env not found at ${DUO_ST_ENV}" >&2
  echo "[error] Run: bash streaming/StreamingTom/scripts/setup_duo_st_env.sh" >&2
  exit 1
fi

export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="${ROOT}:${ROOT}/streaming/StreamingTom:${PYTHONPATH:-}"

# ── config ────────────────────────────────────────────────────────────────────
MODEL=${MODEL:-lmms-lab/llava-onevision-qwen2-0.5b-ov}
SAMPLE_FPS=${SAMPLE_FPS:-0.5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
NUM_GPUS=${NUM_GPUS:-2}
NUM_CHUNKS=${NUM_CHUNKS:-40}
RESUME=${RESUME:-1}
DRY_RUN=${DRY_RUN:-0}

# Space-separated lists — override via env vars
DATASETS=${DATASETS:-"rvs_ego rvs_movie"}
METHODS=${METHODS:-"streamingtom duo_plus_streamingtom"}

OUTPUT_BASE=${OUTPUT_BASE:-${ROOT}/outputs/evaluations_streaming/untracked}
STREAMINGTOM_ROOT=${STREAMINGTOM_ROOT:-streaming/StreamingTom}

DUO_ATTN_DIR=${DUO_ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
DUO_HEADS_FILE=${DUO_HEADS_FILE:-${DUO_ATTN_DIR}/full_attention_heads_latest.tsv}
DUO_THRESHOLD=${DUO_THRESHOLD:-0.5}
DUO_SPARSITY=${DUO_SPARSITY:-0.75}
DUO_SINK_SIZE=${DUO_SINK_SIZE:-256}
DUO_RECENT_SIZE=${DUO_RECENT_SIZE:-512}

# ── validate env ──────────────────────────────────────────────────────────────
echo "[runpod] Root:     ${ROOT}"
echo "[runpod] Datasets: ${DATASETS}"
echo "[runpod] Methods:  ${METHODS}"
echo "[runpod] GPUs:     ${NUM_GPUS}  Chunks/combo: ${NUM_CHUNKS}"
echo "[runpod] Python:   $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── helpers ───────────────────────────────────────────────────────────────────

# Run all chunks for one (dataset, method) combo, batching NUM_GPUS at a time.
run_combo() {
  local dataset=$1 method=$2
  local output_root="${OUTPUT_BASE}/${dataset}/full_eval/run2"
  local log_dir="${output_root}/${method}/logs"
  mkdir -p "${log_dir}"

  echo ""
  echo "────────────────────────────────────────────────────────────"
  echo "[runpod] dataset=${dataset}  method=${method}  chunks=${NUM_CHUNKS}  gpus=${NUM_GPUS}"
  echo "         output: ${output_root}/${method}"
  echo "────────────────────────────────────────────────────────────"

  local chunk batch_pids=() gpu fail=0 batch_start=0

  for (( chunk=0; chunk<NUM_CHUNKS; chunk++ )); do
    gpu=$(( (chunk - batch_start) % NUM_GPUS ))
    local output_path="${output_root}/${method}/chunk_$(printf '%03d' "${chunk}").json"
    local log_path="${log_dir}/chunk_$(printf '%03d' "${chunk}").log"
    mkdir -p "$(dirname "${output_path}")"

    local cmd=(
      python streaming/StreamingTom/scripts/eval/run_eval.py
      --dataset "${dataset}"
      --hf-repo-id Becomebright/RVS
      --allow-hf-video-download
      --model "${MODEL}"
      --method "${method}"
      --sample-fps "${SAMPLE_FPS}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --video-decode-threads 1
      --num-chunks "${NUM_CHUNKS}"
      --chunk-index "${chunk}"
      --output-path "${output_path}"
      --streamingtom-root "${STREAMINGTOM_ROOT}"
    )
    [[ "${RESUME}" == "1" ]] && cmd+=(--resume)
    if [[ "${method}" == "duo_plus_streamingtom" ]]; then
      cmd+=(
        --duo-attn-dir "${DUO_ATTN_DIR}"
        --duo-heads-file "${DUO_HEADS_FILE}"
        --duo-threshold "${DUO_THRESHOLD}"
        --duo-sparsity "${DUO_SPARSITY}"
        --duo-sink-size "${DUO_SINK_SIZE}"
        --duo-recent-size "${DUO_RECENT_SIZE}"
      )
    fi

    echo "[runpod] GPU${gpu} chunk=${chunk}/${NUM_CHUNKS} → ${log_path}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "  DRY_RUN: ${cmd[*]}"
    else
      "${cmd[@]}" > "${log_path}" 2>&1 &
      batch_pids+=($!)
    fi

    # When we've filled all GPUs (or reached the last chunk), wait for this batch
    local next=$(( chunk + 1 ))
    if (( next % NUM_GPUS == 0 )) || (( next == NUM_CHUNKS )); then
      if [[ "${DRY_RUN}" != "1" ]]; then
        for pid in "${batch_pids[@]}"; do
          if wait "${pid}"; then
            echo "[runpod] PID ${pid} done OK"
          else
            echo "[runpod] PID ${pid} FAILED" >&2
            fail=$(( fail + 1 ))
          fi
        done
        batch_pids=()
        batch_start=$(( chunk + 1 ))
      fi
    fi
  done

  if [[ "${DRY_RUN}" != "1" && "${fail}" -gt 0 ]]; then
    echo "[runpod] ${fail} chunk(s) FAILED for ${dataset}/${method}" >&2
    return 1
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    echo "[runpod] Merging chunks for ${dataset}/${method}..."
    python streaming/StreamingTom/scripts/eval/merge_chunks.py \
      --run-root "${output_root}" \
      --methods "${method}"
    echo "[runpod] Merged: ${output_root}/merged/${method}.json"
  fi
}

# ── main loop: dataset × method ───────────────────────────────────────────────
TOTAL_FAIL=0
for dataset in ${DATASETS}; do
  for method in ${METHODS}; do
    run_combo "${dataset}" "${method}" || TOTAL_FAIL=$(( TOTAL_FAIL + 1 ))
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[runpod] DRY_RUN complete — no processes launched"
  exit 0
fi

echo ""
echo "============================================================"
if [[ "${TOTAL_FAIL}" -gt 0 ]]; then
  echo "[runpod] DONE with ${TOTAL_FAIL} combo(s) FAILED" >&2
  exit 1
fi
echo "[runpod] ALL COMBOS COMPLETE"
echo "  Results under: ${OUTPUT_BASE}/<dataset>/full_eval/run2/"
