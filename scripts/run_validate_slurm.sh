#!/usr/bin/env bash
# Quick SLURM job to validate the env backend stack after setup.
# Run after run_setup_slurm.sh completes.
#
#SBATCH --job-name=streaming-validate
#SBATCH --partition=gpunodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${ROOT}/logs"

source /u/navdeep/miniconda3/etc/profile.d/conda.sh

PROJECT_ENV="${ROOT}/envs/duo"
if [[ ! -d "${PROJECT_ENV}" ]]; then
  echo "[validate] ERROR: env not found at ${PROJECT_ENV}" >&2
  echo "[validate] Did setup complete successfully? Check logs/setup_*.log" >&2
  exit 1
fi

conda activate "${PROJECT_ENV}"

echo "[validate] Python: $(which python)"
echo "[validate] GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)"
echo ""

cd "${ROOT}"
python -m streaming.ReKV.validate_runtime_env | tee "${ROOT}/logs/validate_$(date +%Y%m%d_%H%M%S).json"

echo ""
echo "[validate] Key check:"
python -m streaming.ReKV.validate_runtime_env 2>/dev/null | \
  python3 -c "
import json, sys
d = json.load(sys.stdin)
duo = d['method_backend_resolution']['duo_streaming']['backend_resolution']
backend = duo.get('streaming_attn_backend_actual', 'MISSING')
flash = d['method_backend_resolution']['full_streaming']['backend_resolution'].get('flash_attn_available')
cuda = d['torch_runtime']['cuda_available']
print(f'  cuda_available:                {cuda}')
print(f'  flash_attn_available:          {flash}')
print(f'  streaming_attn_backend_actual: {backend}')
if backend == 'blocksparse':
    print('  => PASS: env is ready for paper-faithful evaluation')
else:
    print('  => FAIL: block_sparse_attn not working — check logs/setup_*.log')
    sys.exit(1)
"
