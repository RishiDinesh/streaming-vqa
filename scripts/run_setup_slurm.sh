#!/usr/bin/env bash
# One-shot SLURM job to build the project conda env on a GPU compute node.
# Submit with: bash scripts/run_setup_slurm.sh  (the wrapper handles sbatch)
#
#SBATCH --job-name=streaming-setup
#SBATCH --partition=gpunodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail

# Hardcoded: BASH_SOURCE is unreliable when SLURM copies the script to spool.
ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
mkdir -p "${ROOT}/logs"

source /u/navdeep/miniconda3/etc/profile.d/conda.sh

echo "[setup] hostname: $(hostname)"
echo "[setup] GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo unknown)"
echo "[setup] date: $(date -u)"

cd "${ROOT}"

# sm_86 = RTX A6000/A4000/A4500/A2000, sm_89 = RTX 4090
# Only archs present on this cluster.
BLOCK_SPARSE_ATTN_CUDA_ARCHS="86;89" bash setup.sh

echo "[setup] DONE at $(date -u)"
