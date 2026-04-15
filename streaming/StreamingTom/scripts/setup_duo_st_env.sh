#!/usr/bin/env bash
# Build the duo-st conda environment for StreamingTom methods (5-6).
# Works on RunPod, bare GPU servers, or any machine with conda + CUDA 12.4.
#
# Usage (run directly on the GPU machine — no SLURM needed):
#   bash streaming/StreamingTom/scripts/setup_duo_st_env.sh
#
# On SLURM (must run on a GPU compute node):
#   sbatch streaming/StreamingTom/scripts/setup_duo_st_env.sh
#
# Requirements:
#   - CUDA 12.4 (torch wheel is cu124)
#   - conda or miniconda installed
#   - ~20 GB disk space (torch + flash-attn + LLaVA-NeXT)
#
# After this script: conda activate <repo>/envs/duo-st
#
#SBATCH --job-name=setup-duo-st
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_duo_st_%j.log
#SBATCH --error=logs/setup_duo_st_%j.err

set -euo pipefail

# Resolve repo root: works whether called via sbatch (SLURM copies to spool)
# or directly as `bash setup_duo_st_env.sh` from any directory.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Running under SLURM — ROOT must be set or default to /root/streaming-vqa
  ROOT=${ROOT:-/root/streaming-vqa}
else
  ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
fi

LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "[setup] Repo root: ${ROOT}"
echo "[setup] GPU info:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "(nvidia-smi not available)"

# ── find conda ────────────────────────────────────────────────────────────────
CONDA_INIT_SCRIPT=""
for _candidate in \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh" \
  "/root/miniconda3/etc/profile.d/conda.sh" \
  "/usr/local/miniconda3/etc/profile.d/conda.sh" \
  "/root/miniconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${_candidate}" ]]; then
    CONDA_INIT_SCRIPT="${_candidate}"
    break
  fi
done

if [[ -z "${CONDA_INIT_SCRIPT}" ]]; then
  echo "[setup] conda not found — installing Miniconda to ${HOME}/miniconda3"
  MINICONDA_INSTALLER="/tmp/miniconda_install.sh"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o "${MINICONDA_INSTALLER}"
  bash "${MINICONDA_INSTALLER}" -b -p "${HOME}/miniconda3"
  rm -f "${MINICONDA_INSTALLER}"
  CONDA_INIT_SCRIPT="${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

source "${CONDA_INIT_SCRIPT}"
echo "[setup] conda: $(conda --version)"

# ── create env ────────────────────────────────────────────────────────────────
DUO_ST_ENV="${ROOT}/envs/duo-st"

# Redirect package cache to repo scratch (avoids home quota limits on SLURM)
export CONDA_PKGS_DIRS="${ROOT}/.conda_pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"

if [[ -d "${DUO_ST_ENV}" ]] && [[ -f "${DUO_ST_ENV}/bin/python" ]]; then
  echo "[setup] Env already exists at ${DUO_ST_ENV} — skipping conda create"
else
  echo "[setup] Creating env at ${DUO_ST_ENV} (Python 3.10)"
  CONDA_PKGS_DIRS="${ROOT}/.conda_pkgs" conda create -p "${DUO_ST_ENV}" python=3.10 --solver=classic -y
fi

conda activate "${DUO_ST_ENV}"
echo "[setup] Python: $(which python) ($(python -V))"

# ── pip packages ──────────────────────────────────────────────────────────────
echo "[setup] Upgrading pip"
python -m pip install --upgrade pip

echo "[setup] Installing core dependencies"
python -m pip install --no-cache-dir \
  accelerate==1.9.0 \
  transformers==4.53.3 \
  huggingface-hub==0.36.2 \
  datasets==4.0.0 \
  sacrebleu==2.5.1 \
  einops==0.8.2 \
  av==17.0.0 \
  timm==1.0.26 \
  ftfy==6.3.1 \
  wcwidth==0.6.0 \
  imageio==2.37.3 \
  imageio-ffmpeg==0.6.0 \
  decord==0.6.0 \
  open_clip_torch==3.3.0 \
  tqdm==4.67.3 \
  sentencepiece==0.2.0 \
  protobuf==6.31.1 \
  tensor_parallel==2.0.0 \
  setuptools==80.10.2 \
  bitsandbytes==0.47.0 \
  scipy==1.15.3 \
  scikit-learn==1.7.2 \
  matplotlib \
  numpy==2.2.6 \
  rouge-score

echo "[setup] Installing PyTorch 2.5.1 + CUDA 12.4"
python -m pip install --no-cache-dir --ignore-installed --no-deps \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[setup] Installing flash-attn 2.7.4 (torch2.5/cu12)"
python -m pip install --no-cache-dir \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "[setup] Installing flashinfer 0.2.2 (torch2.5/cu124)"
python -m pip install --no-cache-dir \
  "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.5-cp38-abi3-linux_x86_64.whl"

echo "[setup] Installing LLaVA-NeXT (editable)"
python -m pip install -e "${ROOT}/streaming/StreamingTom/LLaVA-NeXT"

echo "[setup] Installing lmms-eval (editable)"
python -m pip install -e "${ROOT}/streaming/StreamingTom/lmms-eval"

echo "[setup] Installing repo package (duo_attn + streamingtom)"
python -m pip install -e "${ROOT}"

# ── smoke test ────────────────────────────────────────────────────────────────
echo "[setup] Running smoke test"
PYTHONPATH="${ROOT}" python - <<'PY'
import torch, flash_attn, flashinfer
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  available={torch.cuda.is_available()}")
print(f"flash_attn {flash_attn.__version__}")
print(f"flashinfer {flashinfer.__version__}")
import llava; print(f"llava OK")
import duo_attn; print(f"duo_attn OK")
import streamingtom; print(f"streamingtom OK")
print("ALL OK")
PY

echo ""
echo "[setup] Done. Activate with:"
echo "  source ${CONDA_INIT_SCRIPT}"
echo "  conda activate ${DUO_ST_ENV}"
