#!/usr/bin/env bash
# Create the named 'duo' conda env for StreamingTom on a GPU node.
# Submit with: sbatch streaming/StreamingTom/scripts/setup_streamingtom_env.sh
#
#SBATCH --job-name=setup-duo-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa/logs/setup_duo_env_%j.log
#SBATCH --error=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa/logs/setup_duo_env_%j.err

set -euo pipefail

ROOT=/w/nobackup/385/scratch-space/expires-2026-Apr-23/navy/streaming-vqa
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

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

echo "[setup] Conda found: ${CONDA_INIT_SCRIPT}"
echo "[setup] nvidia-smi:"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Use scratch space for the env (home quota is too small for torch)
DUO_ST_ENV="${ROOT}/envs/duo-st"

# Remove any stale partial install in home dir (use rm -rf to avoid conda writing to quota-exceeded home)
if [[ -d "/u/navdeep/miniconda3/envs/duo" ]]; then
  echo "[setup] Removing stale partial env at /u/navdeep/miniconda3/envs/duo"
  rm -rf /u/navdeep/miniconda3/envs/duo
  echo "[setup] Done removing stale env"
fi

# Point conda package/env caches to scratch to avoid home quota
export CONDA_PKGS_DIRS="${ROOT}/.conda_pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"

# Create env in scratch space if it doesn't already exist
if [[ -d "${DUO_ST_ENV}" ]] && [[ -f "${DUO_ST_ENV}/bin/python" ]]; then
  echo "[setup] Env at ${DUO_ST_ENV} already exists"
else
  echo "[setup] Creating env at ${DUO_ST_ENV} with Python 3.10"
  # Use CONDA_PKGS_DIRS to redirect downloads away from home, and classic solver
  # to avoid libmamba plugin issues when home quota is exceeded.
  CONDA_PKGS_DIRS="${ROOT}/.conda_pkgs" conda create -p "${DUO_ST_ENV}" python=3.10 --solver=classic -y
fi

conda activate "${DUO_ST_ENV}"
echo "[setup] Using python: $(which python)"
python -V

echo "[setup] Upgrading pip"
python -m pip install --upgrade pip

echo "[setup] Installing core Python dependencies"
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
  numpy==2.2.6

echo "[setup] Installing CUDA 12.4 PyTorch 2.5.1"
python -m pip install --no-cache-dir --ignore-installed --no-deps \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[setup] Installing flash-attn (torch2.5/cu12, cxx11abiFALSE)"
python -m pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

echo "[setup] Installing flashinfer (torch2.5/cu124 build)"
python -m pip install --no-cache-dir \
  https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.5-cp38-abi3-linux_x86_64.whl

echo "[setup] Installing local editable packages"
python -m pip install -e "${ROOT}/streaming/StreamingTom/LLaVA-NeXT"
python -m pip install -e "${ROOT}/streaming/StreamingTom/lmms-eval"

echo "[setup] Installing duo_attn package"
python -m pip install -e "${ROOT}"

echo "[setup] Running backend smoke test"
PYTHONPATH="${ROOT}:${ROOT}/streaming/StreamingTom" python - <<'PY'
import torch
import flash_attn
import flashinfer
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())
print('flash_attn', flash_attn.__version__)
print('flashinfer', flashinfer.__version__)
import llava
print('llava OK:', llava.__version__ if hasattr(llava, '__version__') else 'installed')
import duo_attn
print('duo_attn OK')
import streamingtom
print('streamingtom OK')
PY

echo "[setup] Done. Named 'duo' env is ready."
