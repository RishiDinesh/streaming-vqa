# Multi-modal DuoAttention (MMDA): Adapting DuoAttention for Efficient Multimodal Long-Context Video Question Answering

Link to original [[paper](https://arxiv.org/abs/2410.10819)] 

## Objective
We adapt the idea of DuoAttention to the multi-modal setting, specifically for video question answering tasks. By identifying retrieval heads that are crucial for processing long contexts and streaming heads that focus on recent tokens and attention sinks, we apply a full KV cache to retrieval heads while using a light-weight, constant-length KV cache for streaming heads. This approach significantly reduce both pre-filling and decoding memory and latency in VLMs, enabling efficient long-context video question answering without compromising accuracy.

## Flowchart
![image](images/duo_atten_crop.png)

## Environment Setup

```bash
conda create -yn duo python=3.10
conda activate duo

# Install CUDA toolkit
conda install -y \
  -c "nvidia/label/cuda-12.4.1" \
  -c conda-forge \
  cuda-toolkit

# Install PyTorch with CUDA 12.4 support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install transformers==4.45.2 opencv-python decord accelerate sentencepiece datasets wandb zstandard matplotlib huggingface_hub==0.25.2 tensor_parallel==2.0.0 ninja packaging

# Install FlashAttention
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.6.3+cu124torch2.4-cp310-cp310-linux_x86_64.whl"

# Install FlashInfer
pip install "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl#sha256=d7605fbe3f14ef7f36e702f627c1f06e5a32495b5ebfe34313c3fb15f3e4eb06"

# Install DuoAttention
pip install -e .
```

### Install LMMs-Eval (eval only)

This project uses a patched fork of `lmms-eval` for evaluation. To setup:

```bash
cd /path/to/streaming-vqa
rm -rf lmms-eval
git clone -b duo_attn https://github.com/RishiDinesh/lmms-eval.git
cd lmms-eval
git checkout 146a2836a3e34347d611a348592f25ac22958589
pip install -e .
cd ..
```

### Install Block Sparse Streaming Attention

This project uses a patched fork of Block Sparse Attention with the required `setup.py` changes already applied. To setup:

```bash
cd /path/to/streaming-vqa
rm -rf Block-Sparse-Attention
git clone -b mm_duo_attn https://github.com/RishiDinesh/Block-Sparse-Attention.git
cd Block-Sparse-Attention
git checkout 8e73b82a47de87dba3110e40c38c35f41c3f5d0d
```

This fork targets the cluster GPUs through `setup.py`.
- 86 corresponds to RTX A2000 / A4000 / A4500 / A5000 / A6000 class,
- 89 corresponds to Ada / RTX 4090. 

Then install it from inside the `Block-Sparse-Attention` directory:

```bash
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="86;89"
export MAX_JOBS=4
python setup.py install
cd ..
```

## Verify Installation
To verify that the installation was successful, you can run the following command in the root directory of the project:
```bash
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
```

---

## Evaluation Setup: StreamingTom vs Duo+StreamingTom (RVS)

Covers the `streamingtom` and `duo_plus_streamingtom` methods evaluated on the RVS benchmark using `streaming/StreamingTom/scripts/eval/run_eval.py`.

### 1. Build the environment

The `duo-st` environment includes all required dependencies: PyTorch 2.5.1, flash-attn 2.7.4, flashinfer 0.2.2, LLaVA-NeXT, and duo_attn.

```bash
# On SLURM (runs on a GPU node):
sbatch streaming/StreamingTom/scripts/setup_streamingtom_env.sh

# Or directly on a GPU machine:
bash streaming/StreamingTom/scripts/setup_streamingtom_env.sh
```

The env is created at `<repo>/envs/duo-st`. Activate with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <repo>/envs/duo-st
export PYTHONPATH="<repo>:<repo>/streaming/StreamingTom"
```

### 2. Run smoke test (1 video)

```bash
sbatch streaming/StreamingTom/scripts/eval/run_smoketest_longest_video.sh
```

Runs both methods on 1 video and prints a pass/fail summary. Outputs:
```
outputs/evaluations_streaming/untracked/smoke_st_longest/<method>/chunk_000.json
```

### 3. Run full evaluation

```bash
sbatch streaming/StreamingTom/scripts/eval/run_streamingtom_eval_slurm_array.sh
```

Results land in:
```
outputs/evaluations_streaming/untracked/<dataset>/<method>/
```

### 4. Merge results and generate graphs

```bash
python streaming/merge_all_results.py --dataset rvs_ego
```

Merged JSONs → `outputs/evaluations_streaming/untracked/rvs-ego/merged/`  
Graphs → `outputs/evaluations_streaming/final_graphs/rvs-ego/plots/`

### Key arguments for `run_eval.py`

| Argument | Description |
|----------|-------------|
| `--method` | `streamingtom` or `duo_plus_streamingtom` |
| `--model` | HF model ID (default: `lmms-lab/llava-onevision-qwen2-0.5b-ov`) |
| `--dataset` | Dataset name (e.g. `rvs_ego`) |
| `--sample-fps` | Frame sampling rate (default: `0.5`) |
| `--max-new-tokens` | Max tokens to generate per answer |
| `--streamingtom-root` | Path to `streaming/StreamingTom` directory |
| `--duo-attn-dir` | *(duo_plus_streamingtom only)* Path to DuoAttention training output |
| `--duo-sparsity` | *(duo_plus_streamingtom only)* Sparsity ratio (default: `0.75`) |
| `--duo-sink-size` | *(duo_plus_streamingtom only)* Sink token budget (default: `256`) |
| `--duo-recent-size` | *(duo_plus_streamingtom only)* Recent token budget (default: `512`) |
