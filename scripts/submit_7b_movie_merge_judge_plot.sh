#!/usr/bin/env bash
# After 7B rvs-movie eval chunks are complete:
#   1. Merge all chunks into merged/ (skipping tt0121765 OOM video for full_streaming)
#   2. Submit judge scoring (SLURM array 0-3, one per method)
#   3. Submit plot job (dependency: afterok all judge jobs)
#
# Usage (from repo root):
#   bash scripts/submit_7b_movie_merge_judge_plot.sh
#
# Prerequisite: job 154092 (chunk_004 tt0167190 + tt0186151) must be complete.

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
cd "${ROOT}"

source scripts/streaming_env.sh
activate_streaming_env

EVAL_DIR="${ROOT}/outputs/evaluations_streaming/rvs-movie/7b_full_eval"
MERGED_DIR="${EVAL_DIR}/merged"
mkdir -p "${MERGED_DIR}"

METHODS=(full_streaming duo_streaming rekv duo_plus_rekv)

echo "=== Step 1: Merge chunks ==="
python3 - << 'PYEOF'
import json, glob, os, sys

ROOT = "/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa"
EVAL_DIR = f"{ROOT}/outputs/evaluations_streaming/rvs-movie/7b_full_eval"
MERGED_DIR = f"{EVAL_DIR}/merged"
os.makedirs(MERGED_DIR, exist_ok=True)

METHODS = ["full_streaming", "duo_streaming", "rekv", "duo_plus_rekv"]

for method in METHODS:
    method_dir = f"{EVAL_DIR}/{method}"
    # Collect all chunk files (including per-video files for full_streaming chunk_004)
    chunk_files = sorted(glob.glob(f"{method_dir}/chunk_*.json"))
    all_videos = []
    run_config = None
    evaluation_manifest = None
    skipped = []
    for cf in chunk_files:
        d = json.load(open(cf))
        if run_config is None:
            run_config = d.get("run_config", {})
        if evaluation_manifest is None:
            evaluation_manifest = d.get("evaluation_manifest", {})
        videos = d.get("videos", [])
        if not videos:
            skipped.append(os.path.basename(cf))
        all_videos.extend(videos)

    out_path = f"{MERGED_DIR}/{method}.json"
    merged = {
        "run_config": run_config,
        "evaluation_manifest": evaluation_manifest,
        "videos": all_videos,
    }
    json.dump(merged, open(out_path, "w"), indent=2)
    print(f"  {method}: {len(all_videos)} videos -> {out_path}"
          + (f"  [skipped empty: {skipped}]" if skipped else ""))

sys.exit(0)
PYEOF

echo ""
echo "=== Step 2: Submit judge scoring (array 0-3) ==="
JUDGE_JIDS=()
for i in 0 1 2 3; do
  JID=$(sbatch \
    --array=${i} \
    --output="logs/7b-movie-judge-${i}-%j.out" \
    --parsable \
    scripts/run_7b_movie_judge_plot.sh)
  echo "  judge task ${i}: job ${JID}"
  JUDGE_JIDS+=("${JID}")
done

# Build dependency string: afterok:j1:j2:j3:j4
DEP=$(IFS=:; echo "afterok:${JUDGE_JIDS[*]}")

echo ""
echo "=== Step 3: Submit plot job (dependency: ${DEP}) ==="
PLOT_JID=$(sbatch \
  --job-name=7b-movie-plot \
  --nodes=1 --ntasks=1 \
  --partition=gpunodes \
  --gres=gpu:rtx_a6000:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:30:00 \
  --output="logs/7b-movie-plot-%j.out" \
  --dependency="${DEP}" \
  --parsable \
  --wrap="
    source ${ROOT}/scripts/streaming_env.sh && activate_streaming_env
    cd ${ROOT}
    MERGED_DIR=${ROOT}/outputs/evaluations_streaming/rvs-movie/7b_full_eval/merged
    python -m streaming.ReKV.plot_results \
      --output-dir \${MERGED_DIR}/plots_judge \
      \${MERGED_DIR}/full_streaming.json \
      \${MERGED_DIR}/duo_streaming.json \
      \${MERGED_DIR}/rekv.json \
      \${MERGED_DIR}/duo_plus_rekv.json
    echo 'Plots written to '\${MERGED_DIR}/plots_judge
  ")
echo "  plot job: ${PLOT_JID}"

echo ""
echo "=== All submitted ==="
echo "Monitor: watch -n 30 'squeue -u navdeep'"
echo "Plots will appear in: ${MERGED_DIR}/plots_judge/"
