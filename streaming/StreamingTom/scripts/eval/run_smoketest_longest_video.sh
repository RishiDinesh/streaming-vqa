#!/usr/bin/env bash
# OOM stress smoke test: both ST methods on the longest rvs-ego video (~60 min, 3611s).
# Only 1 conversation is answered to keep wall time reasonable.
# Verifies full-length ingest (all ~1805 frames at 0.5 fps) without OOM.
#
# Submit with:
#   sbatch streaming/StreamingTom/scripts/eval/run_smoketest_longest_video.sh
#
#SBATCH --job-name=st-smoke-longest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa/logs/st_smoke_longest_%j.log
#SBATCH --error=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa/logs/st_smoke_longest_%j.err

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

export HF_HOME="${ROOT}/.hf_cache"
export TOKENIZERS_PARALLELISM="false"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

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

DUO_ST_ENV="${ROOT}/envs/duo-st"
if [[ -d "${DUO_ST_ENV}" ]]; then
  conda activate "${DUO_ST_ENV}"
else
  conda activate duo
fi
echo "[env] Using python: $(which python)"

cd "${ROOT}"
echo "[smoke] GPU info:"; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Longest rvs-ego video: ~3611s (~60 min), ~1805 frames at 0.5 fps
LONGEST_VIDEO_ID="e58207f1-84ec-424b-a997-ff64a57eb13b"

MODEL=${MODEL:-lmms-lab/llava-onevision-qwen2-0.5b-ov}
DATASET=rvs_ego
SAMPLE_FPS=0.5
MAX_NEW_TOKENS=32
OUTPUT_ROOT="${ROOT}/outputs/evaluations_streaming/smoke_st_longest"
DUO_ATTN_DIR=${DUO_ATTN_DIR:-outputs/train/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632}
DUO_HEADS_FILE="${DUO_ATTN_DIR}/full_attention_heads_latest.tsv"

COMMON=(
  --dataset "${DATASET}"
  --hf-repo-id Becomebright/RVS
  --allow-hf-video-download
  --model "${MODEL}"
  --sample-fps "${SAMPLE_FPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --video-decode-threads 1
  --num-chunks 1 --chunk-index 0
  --video-id "${LONGEST_VIDEO_ID}"
  --max-conversations-per-video 1
  --seed 42
  --streamingtom-root streaming/StreamingTom
)

inspect_result() {
  local f="$1" label="$2"
  if [[ ! -f "${f}" ]]; then
    echo "[FAIL] ${label}: output file missing"; return 1
  fi
  python - "${f}" "${label}" <<'PY'
import json, sys
f, label = sys.argv[1], sys.argv[2]
data = json.load(open(f))
videos = data.get("videos", [])
convs = [c for v in videos for c in v.get("conversations", [])]
agg = data.get("aggregate_metrics", {})
if not videos or not convs:
    print(f"[FAIL] {label}: no videos/conversations"); sys.exit(1)
ms = convs[0].get("method_stats", {})
rt = videos[0].get("runtime_stats", {})
print(f"[OK] {label}")
print(f"     video_id={videos[0].get('video_id')}  duration={videos[0].get('duration')}s")
print(f"     frames_ingested={rt.get('frames_ingested')}  avg_ingest_latency={rt.get('avg_frame_ingest_latency_sec'):.3f}s")
print(f"     conversations={len(convs)}  prediction={convs[0].get('prediction','')[:80]!r}")
print(f"     rouge_l_f1={convs[0].get('scores',{}).get('rouge_l_f1')}")
print(f"     ttft_sec={ms.get('ttft_sec')}  answer_latency_sec={ms.get('answer_latency_sec'):.2f}s")
print(f"     peak_memory_bytes={ms.get('peak_memory_bytes')}  ({(ms.get('peak_memory_bytes') or 0)/1e9:.1f} GB)")
print(f"     retrieval_latency_sec={ms.get('retrieval_latency_sec')}  avg_retrieved_block_count={ms.get('avg_retrieved_block_count')}")
print(f"     cpu_offload_bytes_peak={ms.get('cpu_offload_bytes_peak')}")
PY
}

PASS=0; FAIL=0

# ── method 5: streamingtom ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[smoke 1/2] streamingtom — longest rvs-ego video"
echo "============================================================"
OUT="${OUTPUT_ROOT}/streamingtom/chunk_000.json"
mkdir -p "$(dirname "${OUT}")"
python streaming/StreamingTom/scripts/eval/run_eval.py \
  "${COMMON[@]}" --method streamingtom \
  --output-path "${OUT}" --overwrite-output \
  && RC=0 || RC=$?

if [[ ${RC} -eq 0 ]]; then
  inspect_result "${OUT}" "streamingtom" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
else
  echo "[FAIL] streamingtom exited with code ${RC}"; FAIL=$((FAIL+1))
fi

echo ""
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo ""

# ── method 6: duo_plus_streamingtom ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "[smoke 2/2] duo_plus_streamingtom — longest rvs-ego video"
echo "============================================================"
OUT="${OUTPUT_ROOT}/duo_plus_streamingtom/chunk_000.json"
mkdir -p "$(dirname "${OUT}")"
python streaming/StreamingTom/scripts/eval/run_eval.py \
  "${COMMON[@]}" --method duo_plus_streamingtom \
  --duo-attn-dir "${DUO_ATTN_DIR}" \
  --duo-heads-file "${DUO_HEADS_FILE}" \
  --duo-threshold 0.5 --duo-sparsity 0.75 \
  --duo-sink-size 256 --duo-recent-size 512 \
  --output-path "${OUT}" --overwrite-output \
  && RC=0 || RC=$?

if [[ ${RC} -eq 0 ]]; then
  inspect_result "${OUT}" "duo_plus_streamingtom" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
else
  echo "[FAIL] duo_plus_streamingtom exited with code ${RC}"; FAIL=$((FAIL+1))
fi

echo ""
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo ""
echo "============================================================"
echo "[smoke] SUMMARY: ${PASS} passed / ${FAIL} failed"
echo "============================================================"
[[ ${FAIL} -eq 0 ]] && echo "[smoke] ALL PASSED" || { echo "[smoke] FAILED" >&2; exit 1; }
