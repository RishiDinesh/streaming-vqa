# `benchmark_prefill_chunk_sweep_llava.py`

This benchmark fixes the input context near a target length, then sweeps over several `prefill_chunk_size` values to compare prefill-time behavior between:

- `baseline`
- `duo` (DuoAttention enabled)

The main question it answers is:

How do latency and GPU memory change when we prefill the same long video-text context using different chunk sizes?


## What It Evaluates

The script holds the effective context length roughly constant, then varies only the prefill chunk size.

For each chunk size and each mode, it measures:

- `prefill_total_ms`: total wall-clock time for the full prefill pass
- `ctx_latency`: average milliseconds per prefill chunk
- `ctx_memory`: peak allocated GPU memory during the prefill stage, in MB
- `num_chunks`: how many chunks were needed to cover the full input sequence
- `oom`: whether the run hit CUDA OOM

This makes it useful for answering questions like:

- Does smaller chunking reduce peak memory?
- How much latency overhead does chunking introduce?
- How different are baseline and DuoAttention under the same fixed context?


## High-Level Flow

The benchmark runs in five stages.

### 1. Resolve the prompt and processor

The script loads the prompt text and creates an `AutoProcessor` for the model.

Relevant code:

- [`benchmark_prefill_chunk_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py#L230)
- [`benchmark_context_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_context_sweep_llava.py#L274)

### 2. Calibrate a fixed context length

Before benchmarking chunk sizes, the script finds a number of video frames that produces a sequence length close to `--target_context`.

It does this by:

- choosing a prefix ratio from the requested target context
- sampling frames from the video prefix
- tokenizing the video + prompt
- binary searching over frame count until the tokenized length reaches the target

Relevant code:

- [`benchmark_prefill_chunk_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py#L233)
- [`benchmark_context_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_context_sweep_llava.py#L341)

The output of this calibration step is a `SweepPoint` containing:

- `target_context`
- `actual_context`
- `prefix_seconds`
- `prefix_ratio`
- `num_frames`

### 3. Materialize the fixed input once

After calibration, the benchmark:

- loads the chosen frames from the video
- encodes the video and prompt into a processor batch
- later moves that batch to CUDA
- builds `inputs_embeds` for the model

Relevant code:

- [`benchmark_prefill_chunk_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py#L247)
- [`benchmark_dynamic_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_dynamic_llava.py#L123)

The important design choice here is that the same calibrated input is reused across all chunk-size runs, so the benchmark isolates chunk-size effects instead of changing the input each time.

### 4. Run baseline and DuoAttention

The script loops over:

- `baseline`
- `duo`

Model setup differs by mode:

- `baseline` enables tuple KV cache
- `duo` loads an attention pattern from `--attn_load_dir`, sparsifies it, and enables DuoAttention evaluation mode

Relevant code:

- [`benchmark_prefill_chunk_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py#L260)
- [`benchmark_context_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_context_sweep_llava.py#L425)

If `--attn_load_dir` is omitted, the script skips `duo` mode and only runs `baseline`.

### 5. Sweep prefill chunk size

For each value in `--prefill_chunk_sizes`, the benchmark:

- splits `inputs_embeds` along sequence length
- feeds chunks sequentially into `model.language_model(...)`
- threads `past_key_values` from one chunk to the next
- measures total CUDA event time for the whole prefill loop
- records peak allocated GPU memory during that loop

Relevant code:

- [`benchmark_prefill_chunk_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py#L70)

This is a prefill-only benchmark. It does not benchmark decode steps.


## Metric Definitions

### `prefill_total_ms`

Total elapsed CUDA time for the full prefill pass across all chunks.

If the sequence is length `32000` and chunk size is `4000`, the benchmark runs 8 chunks and reports the total time across all 8.

### `ctx_latency`

Average milliseconds per prefill chunk:

`prefill_total_ms / num_chunks`

This is useful for seeing whether larger chunks make each chunk slower, even if the total number of chunks decreases.

### `ctx_memory`

Peak allocated GPU memory during the prefill stage, measured in MB.

It is computed from:

`torch.cuda.max_memory_allocated() / 1024 / 1024`

Relevant code:

- [`benchmark_prefill_chunk_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py#L75)
- [`benchmark_prefill_chunk_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py#L102)

Important nuance:

- this is allocated memory, not reserved memory
- this is the peak over the whole prefill loop, not per-chunk memory
- this is GPU memory only

### `num_chunks`

The number of chunks used to cover the full sequence:

`ceil(seq_len / prefill_chunk_size)`

### `oom`

Whether the run threw a CUDA out-of-memory error and was caught by the benchmark.

Relevant code:

- [`benchmark_prefill_chunk_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py#L112)


## Output Files

The benchmark writes the following files into `--output_dir`:

- `prefill_chunk_sweep_results.json`
- `prefill_chunk_sweep_results.csv`
- `prefill_chunk_sweep_summary.txt`
- `prefill_chunk_sweep_config.json`
- `prefill_chunk_sweep_plot.png` unless `--skip_plot` is used

The JSON rows contain one record per:

- mode
- chunk size

Typical row fields:

- `mode`
- `target_context`
- `actual_context`
- `prefix_seconds`
- `prefix_ratio`
- `num_frames`
- `sparsity`
- `prefill_chunk_size`
- `ctx_latency`
- `ctx_memory`
- `prefill_total_ms`
- `num_chunks`
- `oom`


## Example Command

From the repository root:

```bash
python -u duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py \
  --model_name /root/streaming-vqa/models/llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --video_path data/samplee.mp4 \
  --output_dir /root/streaming-vqa/untracked/prefill_chunk_sweep_32k \
  --attn_load_dir /root/streaming-vqa/outputs/0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles1 \
  --prompt_file /root/streaming-vqa/long_prompt.txt \
  --target_context 32000 \
  --max_length 32000 \
  --prefill_chunk_sizes 4000 8000 12000 16000 20000 24000 28000 32000
```

To run baseline only, remove `--attn_load_dir`.


## How To Read the Results

In general:

- smaller chunk sizes usually lower peak memory
- smaller chunk sizes may increase total prefill time because they add more chunk iterations
- larger chunk sizes reduce chunk count but can sharply increase memory
- DuoAttention should usually use much less memory than baseline at the same context and chunk size

When comparing results, check these fields together:

- `prefill_total_ms` for end-to-end prefill cost
- `ctx_memory` for memory pressure
- `num_chunks` to understand the chunking regime

If a large chunk size shows much worse memory but only modest latency benefit, it is often not worth using.


## Related Scripts

- [`benchmark_prefill_chunk_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_prefill_chunk_sweep_llava.py)
- [`plot_prefill_chunk_sweep.py`](/root/streaming-vqa/plot_prefill_chunk_sweep.py)
- [`plot_prefill_chunk_sweep_32k.py`](/root/streaming-vqa/plot_prefill_chunk_sweep_32k.py)
- [`benchmark_context_sweep_llava.py`](/root/streaming-vqa/duo_attn/eval/efficiency/benchmark_context_sweep_llava.py)
