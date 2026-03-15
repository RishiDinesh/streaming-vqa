#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="duo"

echo "==> Python version"
python -V


echo "==> Installing CUDA toolkit"
conda install -y \
  -c "nvidia/label/cuda-12.4.1" \
  -c conda-forge \
  cuda-toolkit

echo "==> Verifying nvcc"
nvcc --version

echo "==> Installing Torch 2.4.1 +Cu12.4"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

echo "==> Verifying PyTorch"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

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
  packaging

echo "==> Installing FlashAttention wheel"
pip install \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu124torch2.4-cp310-cp310-linux_x86_64.whl"

echo "==> Verifying FlashAttention"
python - <<'PY'
import flash_attn, torch
print("flash_attn ok")
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
PY

echo "==> Installing FlashInfer"
pip install "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl#sha256=d7605fbe3f14ef7f36e702f627c1f06e5a32495b5ebfe34313c3fb15f3e4eb06"

echo "==> Verifying FlashInfer"
python - <<'PY'
import flashinfer
print("flashinfer ok")
PY

echo "==> Installing DuoAttention in editable mode"
pip install -e .

echo "==> Cloning and installing Block-Sparse-Attention"
if [ ! -d "Block-Sparse-Attention" ]; then
  git clone https://github.com/mit-han-lab/Block-Sparse-Attention
fi

pushd Block-Sparse-Attention >/dev/null
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-86;89}"
export MAX_JOBS="${MAX_JOBS:-4}"
echo "==> BLOCK_SPARSE_ATTN_CUDA_ARCHS=${BLOCK_SPARSE_ATTN_CUDA_ARCHS}"
echo "==> MAX_JOBS=${MAX_JOBS}"
python setup.py install
popd >/dev/null

echo "==> Verifying full installation"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
import flash_attn
print("flash_attn ok")
import flashinfer
print("flashinfer ok")
import block_sparse_attn
import block_sparse_attn_cuda
print("block_sparse_attn:", block_sparse_attn.__file__)
print("block_sparse_attn_cuda:", block_sparse_attn_cuda.__file__)
print("block_sparse_attn ok")
PY
