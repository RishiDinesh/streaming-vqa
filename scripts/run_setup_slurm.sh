#!/usr/bin/env bash
# One-shot SLURM job to build the project conda env on a GPU compute node.
# Submits itself — do not sbatch this directly, just: bash scripts/run_setup_slurm.sh
#
# After this job completes, run:
#   bash scripts/run_validate_slurm.sh
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

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${ROOT}/logs"

source /u/navdeep/miniconda3/etc/profile.d/conda.sh

cd "${ROOT}"

echo "[setup] hostname: $(hostname)"
echo "[setup] GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "[setup] date: $(date -u)"

# sm_86 = RTX A6000/A4000/A4500/A2000, sm_89 = RTX 4090
# Only build for the archs this cluster actually has.
BLOCK_SPARSE_ATTN_CUDA_ARCHS="86;89" \
  bash setup.sh 2>&1 | tee "${ROOT}/logs/setup_$(date +%Y%m%d_%H%M%S).log"

echo "[setup] DONE at $(date -u)"
