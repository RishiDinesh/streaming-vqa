#!/usr/bin/env bash
# Unified smoke test for all 6 streaming video-QA methods.
# Tests 1 video, 1 conversation per method with the 0.5B model.
#
# Methods 1-4 (ReKV suite) run under envs/duo  (ReKV + block_sparse_attn env).
# Methods 5-6 (StreamingTom suite) run under envs/duo-st (torch 2.5.1/LLaVA-NeXT env).
#
# Submit with sbatch:
#   sbatch streaming/run_smoke_all_methods.sh
#
# Or run interactively on a GPU node:
#   bash streaming/run_smoke_all_methods.sh
#
#SBATCH --job-name=smoke-all-6-methods
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa/logs/smoke_all_%j.log
#SBATCH --error=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa/logs/smoke_all_%j.err

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"

# ── conda init ────────────────────────────────────────────────────────────────
CONDA_INIT_SCRIPT=""
for _candidate in \
  "/u/navdeep/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "/root/miniconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${_candidate}" ]]; then
    CONDA_INIT_SCRIPT="${_candidate}"
    break
  fi
done
if [[ -z "${CONDA_INIT_SCRIPT}" ]]; then
  echo "[error] Cannot find conda init script" >&2; exit 1
fi
source "${CONDA_INIT_SCRIPT}"

cd "${ROOT}"

# ── shared config ─────────────────────────────────────────────────────────────
MODEL_REKV=${MODEL_REKV:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}
MODEL_ST=${MODEL_ST:-lmms-lab/llava-onevision-qwen2-0.5b-ov}
DATASET=${DATASET:-rvs_ego}
SAMPLE_FPS=0.5
MAX_NEW_TOKENS=16
OUTPUT_ROOT="${ROOT}/outputs/evaluations_streaming/untracked/smoke_all_methods"
DUO_ATTN_DIR=${DUO_ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
DUO_HEADS_FILE="${DUO_ATTN_DIR}/full_attention_heads_latest.tsv"

PASS=0
FAIL=0
RESULTS=()

echo "============================================================"
echo "[smoke] GPU info:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

# ── helper: inspect result JSON ───────────────────────────────────────────────
inspect_result() {
  local f="$1" label="$2"
  if [[ ! -f "${f}" ]]; then
    echo "  [FAIL] ${label}: output file missing: ${f}"
    FAIL=$(( FAIL + 1 ))
    RESULTS+=("FAIL:${label}")
    return
  fi
  python - "${f}" "${label}" <<'PY'
import json, sys
f, label = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(f))
except Exception as e:
    print(f"  [FAIL] {label}: JSON parse error: {e}")
    sys.exit(1)
videos = data.get("videos", [])
convs = [c for v in videos for c in v.get("conversations", [])]
agg = data.get("aggregate_metrics", {})
if not videos:
    print(f"  [FAIL] {label}: no videos in output")
    sys.exit(1)
if not convs:
    print(f"  [FAIL] {label}: no conversations in output")
    sys.exit(1)
ms = convs[0].get("method_stats", {})
print(f"  [OK]   {label}")
print(f"         videos={len(videos)} conversations={len(convs)}")
print(f"         avg_rouge_l_f1={agg.get('avg_rouge_l_f1')}  avg_answer_latency_sec={agg.get('avg_answer_latency_sec')}")
print(f"         peak_memory_bytes={agg.get('peak_memory_bytes')}")
print(f"         ttft_sec={ms.get('ttft_sec')}  retrieval_latency_sec={ms.get('retrieval_latency_sec')}")
print(f"         avg_retrieved_block_count={ms.get('avg_retrieved_block_count')}  cpu_offload_bytes_peak={ms.get('cpu_offload_bytes_peak')}")
# Verify canonical schema fields exist in method_stats
for key in ("retrieval_latency_sec", "avg_retrieved_block_count", "cpu_offload_bytes_current", "cpu_offload_bytes_peak"):
    if key not in ms:
        print(f"  [WARN] {label}: method_stats missing expected key: {key}")
PY
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    PASS=$(( PASS + 1 ))
    RESULTS+=("OK:${label}")
  else
    FAIL=$(( FAIL + 1 ))
    RESULTS+=("FAIL:${label}")
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# METHODS 1–4: ReKV suite (envs/duo)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "== Activating ReKV env (envs/duo) =="
conda activate "${ROOT}/envs/duo"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
echo "[env] Python: $(which python)"

COMMON_REKV=(
  --dataset "${DATASET}"
  --hf-repo-id Becomebright/RVS
  --allow-hf-video-download
  --model "${MODEL_REKV}"
  --sample-fps "${SAMPLE_FPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --video-decode-threads 1
  --num-chunks 1 --chunk-index 0
  --max-videos 1 --max-conversations-per-video 1
  --seed 42
)

# 1. full_streaming
echo ""
echo "============================================================"
echo "[smoke 1/6] full_streaming"
echo "============================================================"
OUT="${OUTPUT_ROOT}/full_streaming/chunk_000.json"
mkdir -p "$(dirname "${OUT}")"
python -m streaming.ReKV.run_eval \
  "${COMMON_REKV[@]}" --method full_streaming \
  --output-path "${OUT}" --overwrite-output
inspect_result "${OUT}" "full_streaming"

# 2. duo_streaming
echo ""
echo "============================================================"
echo "[smoke 2/6] duo_streaming"
echo "============================================================"
OUT="${OUTPUT_ROOT}/duo_streaming/chunk_000.json"
mkdir -p "$(dirname "${OUT}")"
python -m streaming.ReKV.run_eval \
  "${COMMON_REKV[@]}" --method duo_streaming \
  --attn-dir "${DUO_ATTN_DIR}" \
  --sparsity 0.75 --deploy-sink-size 256 --deploy-recent-size 512 \
  --output-path "${OUT}" --overwrite-output
inspect_result "${OUT}" "duo_streaming"

# 3. rekv
echo ""
echo "============================================================"
echo "[smoke 3/6] rekv"
echo "============================================================"
OUT="${OUTPUT_ROOT}/rekv/chunk_000.json"
mkdir -p "$(dirname "${OUT}")"
python -m streaming.ReKV.run_eval \
  "${COMMON_REKV[@]}" --method rekv \
  --n-local 15000 --retrieve-size 64 \
  --output-path "${OUT}" --overwrite-output
inspect_result "${OUT}" "rekv"

# 4. duo_plus_rekv
echo ""
echo "============================================================"
echo "[smoke 4/6] duo_plus_rekv"
echo "============================================================"
OUT="${OUTPUT_ROOT}/duo_plus_rekv/chunk_000.json"
mkdir -p "$(dirname "${OUT}")"
python -m streaming.ReKV.run_eval \
  "${COMMON_REKV[@]}" --method duo_plus_rekv \
  --attn-dir "${DUO_ATTN_DIR}" \
  --sparsity 0.75 --deploy-sink-size 256 --deploy-recent-size 512 \
  --n-local 15000 --retrieve-size 64 \
  --output-path "${OUT}" --overwrite-output
inspect_result "${OUT}" "duo_plus_rekv"

conda deactivate || true

# ─────────────────────────────────────────────────────────────────────────────
# METHODS 5–6: StreamingTom suite (envs/duo-st)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "== Activating StreamingTom env (envs/duo-st) =="
DUO_ST_ENV="${ROOT}/envs/duo-st"
if [[ -d "${DUO_ST_ENV}" ]]; then
  conda activate "${DUO_ST_ENV}"
else
  conda activate duo
fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
echo "[env] Python: $(which python)"

COMMON_ST=(
  --dataset "${DATASET}"
  --hf-repo-id Becomebright/RVS
  --allow-hf-video-download
  --model "${MODEL_ST}"
  --sample-fps "${SAMPLE_FPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --video-decode-threads 1
  --num-chunks 1 --chunk-index 0
  --max-videos 1 --max-conversations-per-video 1
  --seed 42
  --streamingtom-root streaming/StreamingTom
)

# 5. streamingtom
echo ""
echo "============================================================"
echo "[smoke 5/6] streamingtom"
echo "============================================================"
OUT="${OUTPUT_ROOT}/streamingtom/chunk_000.json"
mkdir -p "$(dirname "${OUT}")"
python streaming/StreamingTom/scripts/eval/run_eval.py \
  "${COMMON_ST[@]}" --method streamingtom \
  --output-path "${OUT}" --overwrite-output
inspect_result "${OUT}" "streamingtom"

# 6. duo_plus_streamingtom
echo ""
echo "============================================================"
echo "[smoke 6/6] duo_plus_streamingtom"
echo "============================================================"
OUT="${OUTPUT_ROOT}/duo_plus_streamingtom/chunk_000.json"
mkdir -p "$(dirname "${OUT}")"
python streaming/StreamingTom/scripts/eval/run_eval.py \
  "${COMMON_ST[@]}" --method duo_plus_streamingtom \
  --duo-attn-dir "${DUO_ATTN_DIR}" \
  --duo-heads-file "${DUO_HEADS_FILE}" \
  --duo-threshold 0.5 --duo-sparsity 0.75 \
  --duo-sink-size 256 --duo-recent-size 512 \
  --output-path "${OUT}" --overwrite-output
inspect_result "${OUT}" "duo_plus_streamingtom"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[smoke] SUMMARY: ${PASS} passed / ${FAIL} failed"
for r in "${RESULTS[@]}"; do
  echo "  ${r}"
done
echo "============================================================"

if [[ ${FAIL} -gt 0 ]]; then
  echo "[smoke] FAILED" >&2
  exit 1
fi
echo "[smoke] ALL PASSED"
