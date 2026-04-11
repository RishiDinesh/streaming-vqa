# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG STREAMING_VQA_REPO="https://github.com/RishiDinesh/streaming-vqa.git"
ARG STREAMING_VQA_REF="main"
ARG BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;86;89;90"
ARG MAX_JOBS=4
ARG LMMS_EVAL_REPO="https://github.com/RishiDinesh/lmms-eval.git"
ARG LMMS_EVAL_REF="146a2836a3e34347d611a348592f25ac22958589"
ARG BLOCK_SPARSE_ATTN_REPO="https://github.com/RishiDinesh/Block-Sparse-Attention.git"
ARG BLOCK_SPARSE_ATTN_REF="8e73b82a47de87dba3110e40c38c35f41c3f5d0d"

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BLOCK_SPARSE_ATTN_CUDA_ARCHS=${BLOCK_SPARSE_ATTN_CUDA_ARCHS} \
    MAX_JOBS=${MAX_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    ninja-build \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    curl \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
 && pip install --upgrade pip setuptools wheel

COPY utils/verify_install.py /opt/mmda/verify_install.py

RUN git clone "${STREAMING_VQA_REPO}" /workspace \
 && git -C /workspace -c advice.detachedHead=false checkout "${STREAMING_VQA_REF}"

WORKDIR /workspace

# Preserve the main repo as an editable git checkout for development, while
# replacing the heavy submodule-style deps with fresh pinned clones.
RUN rm -rf /workspace/Block-Sparse-Attention /workspace/lmms-eval /workspace/flash-attention \
 && git clone --branch duo_attn --single-branch "${LMMS_EVAL_REPO}" /workspace/lmms-eval \
 && git -C /workspace/lmms-eval -c advice.detachedHead=false checkout "${LMMS_EVAL_REF}" \
 && git clone --branch mm_duo_attn --single-branch "${BLOCK_SPARSE_ATTN_REPO}" /workspace/Block-Sparse-Attention \
 && git -C /workspace/Block-Sparse-Attention -c advice.detachedHead=false checkout "${BLOCK_SPARSE_ATTN_REF}"

RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1

RUN pip install \
    transformers==4.45.2 \
    opencv-python \
    decord \
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

RUN pip install \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu124torch2.4-cp310-cp310-linux_x86_64.whl"

RUN pip install \
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl#sha256=d7605fbe3f14ef7f36e702f627c1f06e5a32495b5ebfe34313c3fb15f3e4eb06"

RUN pip install -e /workspace/lmms-eval \
 && pip install -e /workspace

RUN cd /workspace/Block-Sparse-Attention \
 && python setup.py install

RUN python /opt/mmda/verify_install.py \
 && python -m duo_attn.train --help >/dev/null \


FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/mmda /opt/mmda
COPY --from=builder /workspace /workspace

RUN mkdir -p \
    /workspace/datasets \
    /workspace/models \
    /workspace/outputs \
    /cache/huggingface \
    /cache/torch

WORKDIR /workspace

ENTRYPOINT ["bash"]
