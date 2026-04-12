#!/usr/bin/env bash
# Environment setup for the streaming-vqa project.
# Installs the 'duo' conda environment into the project directory so the env
# lives in scratch space alongside the code, not in your home directory.
#
# Usage (from the repo root):
#   bash setup.sh
#
# The conda env will be created at:
#   <repo-root>/envs/duo
#
# Subsequent activation (done automatically by streaming_env.sh):
#   conda activate <repo-root>/envs/duo
#
# NVIDIA CUDA targets (set before running):
#   BLOCK_SPARSE_ATTN_CUDA_ARCHS="80"     # A100
#   BLOCK_SPARSE_ATTN_CUDA_ARCHS="89"     # L40 / Ada
#   BLOCK_SPARSE_ATTN_CUDA_ARCHS="90"     # H100
#   BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;89;90" # multi-arch
# Default is "80;89;90".

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_PREFIX="${REPO_ROOT}/envs/duo"

# Redirect conda package cache to scratch space so CUDA toolkit packages
# (~8GB) don't exceed the home-directory quota on /u/navdeep.
export CONDA_PKGS_DIRS="${REPO_ROOT}/.conda_pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"

echo "==> Repo root: ${REPO_ROOT}"
echo "==> Conda env prefix: ${ENV_PREFIX}"
echo "==> Conda pkg cache:  ${CONDA_PKGS_DIRS}"

# ---------------------------------------------------------------------------
# Step 1: create the conda environment (prefix-based, in project dir)
# ---------------------------------------------------------------------------
if [[ -d "${ENV_PREFIX}" ]]; then
  echo "==> Env already exists at ${ENV_PREFIX}; skipping conda create"
else
  echo "==> Creating conda env at ${ENV_PREFIX}"
  conda create -y --prefix "${ENV_PREFIX}" python=3.10
fi

# Activate by prefix
eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX}"
echo "==> Activated: $(which python)"

# ---------------------------------------------------------------------------
# Step 2: Install CUDA toolkit into env (needed for nvcc at build time)
# ---------------------------------------------------------------------------
echo "==> Installing CUDA toolkit 12.4"
conda install -y \
  -c "nvidia/label/cuda-12.4.1" \
  -c conda-forge \
  cuda-toolkit

echo "==> nvcc version"
nvcc --version

# ---------------------------------------------------------------------------
# Step 3: Install PyTorch (CUDA 12.4 wheel)
# ---------------------------------------------------------------------------
echo "==> Installing PyTorch 2.4.1 + CUDA 12.4"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

# ---------------------------------------------------------------------------
# Step 4: Core Python dependencies
# ---------------------------------------------------------------------------
echo "==> Installing Python dependencies"
pip install \
  transformers==4.45.2 \
  accelerate \
  sentencepiece \
  datasets \
  wandb \
  zstandard \
  matplotlib \
  huggingface_hub==0.25.2 \
  tensor_parallel==2.0.0 \
  ninja \
  packaging \
  tqdm \
  numpy \
  imageio \
  imageio-ffmpeg \
  decord \
  rouge_score

# ---------------------------------------------------------------------------
# Step 5: FlashAttention (prebuilt wheel — CUDA 12.4 / torch 2.4 / py3.10)
# ---------------------------------------------------------------------------
echo "==> Installing FlashAttention"
pip install \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu124torch2.4-cp310-cp310-linux_x86_64.whl"

python - <<'PY'
import flash_attn
print("flash_attn ok:", flash_attn.__version__)
PY

# ---------------------------------------------------------------------------
# Step 6: FlashInfer (optional — RMSNorm/RoPE acceleration)
# ---------------------------------------------------------------------------
echo "==> Installing FlashInfer"
pip install "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl#sha256=d7605fbe3f14ef7f36e702f627c1f06e5a32495b5ebfe34313c3fb15f3e4eb06" \
  || echo "[warn] FlashInfer install failed — will use standard RMSNorm/RoPE paths"

# ---------------------------------------------------------------------------
# Step 7: Install this repo in editable mode
# ---------------------------------------------------------------------------
echo "==> Installing duo_attn package (editable)"
pip install -e "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Step 8: Block-Sparse-Attention (required for paper-faithful DuoAttention)
# ---------------------------------------------------------------------------
BLOCK_SPARSE_ATTN_CUDA_ARCHS="${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-80;89;90}"
MAX_JOBS="${MAX_JOBS:-4}"
BSA_DIR="${REPO_ROOT}/Block-Sparse-Attention"

echo "==> Block-Sparse-Attention"
echo "    CUDA archs: ${BLOCK_SPARSE_ATTN_CUDA_ARCHS}"
echo "    MAX_JOBS:   ${MAX_JOBS}"

if [[ ! -d "${BSA_DIR}" ]]; then
  git clone https://github.com/mit-han-lab/Block-Sparse-Attention "${BSA_DIR}"
fi

pushd "${BSA_DIR}" >/dev/null
export BLOCK_SPARSE_ATTN_CUDA_ARCHS
export MAX_JOBS
python setup.py install
popd >/dev/null

# ---------------------------------------------------------------------------
# Step 9: Final verification
# ---------------------------------------------------------------------------
echo "==> Verifying full installation"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| available:", torch.cuda.is_available())
import flash_attn; print("flash_attn ok:", flash_attn.__version__)
try:
    import flashinfer; print("flashinfer ok")
except ImportError:
    print("flashinfer: not installed (optional)")
import block_sparse_attn, block_sparse_attn_cuda
print("block_sparse_attn ok:", block_sparse_attn.__file__)
import decord; print("decord ok:", decord.__version__)
import rouge_score; print("rouge_score ok")
PY

echo ""
echo "==> Setup complete."
echo "    Env prefix: ${ENV_PREFIX}"
echo "    To activate manually: conda activate ${ENV_PREFIX}"
echo "    The streaming scripts activate this env automatically via scripts/streaming_env.sh"
