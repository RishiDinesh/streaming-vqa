# Prefill Metric Flowcharts

This note explains what the previous benchmarking scripts were doing and what the updated scripts do now.

The two relevant entry points are:

- `scripts/prefill_eval/context_length.sh`
- `scripts/prefill_eval/prefil_chunk.sh`

## Previous Behavior

Previously, the scripts reported raw prefill latency and raw peak memory.

### `context_length.sh` before

- The script swept over different context lengths.
- For each context length, it built one video-text prompt near the target token count.
- It benchmarked one full prefill pass over the whole prompt.
- It stored:
  - `ctx_latency`: average milliseconds for one full prefill run
  - `ctx_memory`: peak GPU memory in MB during prefill
- The plot showed absolute latency and absolute memory.

```mermaid
flowchart TD
    A[Run context_length.sh] --> B[Calibrate frames for each target context]
    B --> C[Build video + text prompt]
    C --> D[Create input embeddings]
    D --> E[Benchmark full prefill pass]
    E --> F[Measure ctx_latency]
    E --> G[Measure ctx_memory]
    F --> H[Save raw results JSON/CSV/summary]
    G --> H
    H --> I[Plot absolute latency vs context length]
    H --> J[Plot absolute memory vs context length]
```

### `prefil_chunk.sh` before

- The script first locked a single context length.
- Then it swept over several `prefill_chunk_size` values.
- For each chunk size, it split the full prompt into chunks and prefills chunk-by-chunk.
- It stored:
  - `prefill_total_ms`: total prefill time across all chunks
  - `ctx_latency`: average milliseconds per chunk
  - `ctx_memory`: peak GPU memory in MB during the prefill loop
- The plot mainly showed absolute totals and absolute memory.

```mermaid
flowchart TD
    A[Run prefil_chunk.sh] --> B[Calibrate one fixed context]
    B --> C[Build video + text prompt]
    C --> D[Create input embeddings]
    D --> E[Sweep over chunk sizes]
    E --> F[Split embeddings into chunks]
    F --> G[Run prefill loop chunk by chunk]
    G --> H[Measure prefill_total_ms]
    G --> I[Compute ctx_latency = prefill_total_ms / num_chunks]
    G --> J[Measure ctx_memory]
    H --> K[Save raw results JSON/CSV/summary]
    I --> K
    J --> K
    K --> L[Plot chunk-size results with raw latency/memory]
```

## New Behavior

Now the scripts still save the raw metrics, but they also derive per-token metrics from them.

## Updated Metric Definitions

### For `context_length.sh`

- Raw metric:
  - `ctx_latency` = average milliseconds for one full prefill run
- New derived metric:
  - `prefill_latency_ms_per_token = ctx_latency / actual_context`
  - `prefill_memory_mb_per_token = ctx_memory / actual_context`

### For `prefil_chunk.sh`

- Raw metrics:
  - `prefill_total_ms` = total prefill time across all chunks
  - `ctx_latency` = average milliseconds per chunk
- New derived metric:
  - `prefill_latency_ms_per_token = prefill_total_ms / actual_context`
  - `prefill_memory_mb_per_token = ctx_memory / actual_context`

## Current Flow

### `context_length.sh` now

- The script still sweeps context length as before.
- It still measures raw prefill latency and raw peak memory.
- Then it derives per-token values before saving and plotting.
- The plot now uses:
  - per-token latency
  - per-token normalized memory

```mermaid
flowchart TD
    A[Run context_length.sh] --> B[Calibrate frames for each target context]
    B --> C[Build video + text prompt]
    C --> D[Create input embeddings]
    D --> E[Benchmark full prefill pass]
    E --> F[Measure ctx_latency]
    E --> G[Measure ctx_memory]
    F --> H[Compute prefill_latency_ms_per_token = ctx_latency / actual_context]
    G --> I[Compute prefill_memory_mb_per_token = ctx_memory / actual_context]
    H --> J[Save raw + per-token results]
    I --> J
    J --> K[Plot per-token latency vs context length]
    J --> L[Plot per-token memory vs context length]
```

### `prefil_chunk.sh` now

- The script still locks one context and sweeps chunk size as before.
- It still measures total prefill time, average per chunk, and peak memory.
- Then it derives per-token values before saving and plotting.
- The plot now uses:
  - per-token prefill latency
  - total prefill time
  - peak prefill memory

```mermaid
flowchart TD
    A[Run prefil_chunk.sh] --> B[Calibrate one fixed context]
    B --> C[Build video + text prompt]
    C --> D[Create input embeddings]
    D --> E[Sweep over chunk sizes]
    E --> F[Split embeddings into chunks]
    F --> G[Run prefill loop chunk by chunk]
    G --> H[Measure prefill_total_ms]
    G --> I[Measure ctx_latency per chunk]
    G --> J[Measure ctx_memory]
    H --> K[Compute prefill_latency_ms_per_token = prefill_total_ms / actual_context]
    J --> L[Compute prefill_memory_mb_per_token = ctx_memory / actual_context]
    I --> M[Keep per-chunk latency for reference]
    K --> N[Save raw + per-token results]
    L --> N
    M --> N
    N --> O[Plot per-token latency vs chunk size]
    N --> P[Plot total prefill time vs chunk size]
    N --> Q[Plot peak memory vs chunk size]
```

## Short Summary

- Before: the plots used raw latency and raw peak memory.
- Now: the scripts still keep the raw metrics, but they derive and plot per-token metrics where appropriate.
- This makes comparisons across different context lengths more meaningful, because the values are normalized by token count.
