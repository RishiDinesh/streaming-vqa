#!/usr/bin/env bash
set -euo pipefail

GPU_BACKEND=${GPU_BACKEND:-auto}

echo "==> Python version"
python -V

echo "==> Installing shared Python dependencies"
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
  opencv-python \
  decord

if [[ "${GPU_BACKEND}" == "auto" ]]; then
  if [[ -n "${ROCM_HOME:-}" ]] || [[ -n "${HIP_PATH:-}" ]]; then
    GPU_BACKEND="rocm"
  else
    GPU_BACKEND="cuda"
  fi
fi

if [[ "${GPU_BACKEND}" == "cuda" ]]; then
  echo "==> Installing CUDA PyTorch stack"
  conda install -y \
    -c "nvidia/label/cuda-12.4.1" \
    -c conda-forge \
    cuda-toolkit
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

  echo "==> Installing CUDA-only attention accelerators"
  pip install \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu124torch2.4-cp310-cp310-linux_x86_64.whl"
  pip install \
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl#sha256=d7605fbe3f14ef7f36e702f627c1f06e5a32495b5ebfe34313c3fb15f3e4eb06"

  if [ ! -d "Block-Sparse-Attention" ]; then
    git clone https://github.com/mit-han-lab/Block-Sparse-Attention
  fi

  pushd Block-Sparse-Attention >/dev/null
  export BLOCK_SPARSE_ATTN_CUDA_ARCHS="${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-86;89}"
  export MAX_JOBS="${MAX_JOBS:-4}"
  python setup.py install
  popd >/dev/null
else
  echo "==> Installing ROCm PyTorch stack"
  pip install torch torchvision torchaudio
  echo "==> Skipping flashinfer and Block-Sparse-Attention on ROCm."
  echo "==> DuoAttention will fall back to SDPA kernels when CUDA-only libraries are unavailable."
fi

echo "==> Installing DuoAttention in editable mode"
pip install -e .

echo "==> Verifying installation"
GPU_BACKEND="${GPU_BACKEND}" python - <<'PY'
import importlib.util
import os
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch hip:", getattr(torch.version, "hip", None))
print("cuda available:", torch.cuda.is_available())

for mod in ["flash_attn", "flashinfer", "block_sparse_attn"]:
    print(f"{mod} installed:", importlib.util.find_spec(mod) is not None)

if os.environ.get("GPU_BACKEND") == "rocm" and getattr(torch.version, "hip", None) is None:
    raise SystemExit(
        "Expected a ROCm build of PyTorch, but torch.version.hip is None. "
        "Install the matching ROCm wheel from the official PyTorch/AMD instructions "
        "for your ROCm version, then rerun setup."
    )
PY
