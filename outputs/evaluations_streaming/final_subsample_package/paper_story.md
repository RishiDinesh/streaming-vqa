# Current Paper Story

## Scope
This package summarizes the validated subsample-only streaming evaluation on one MI300X using LLaVA-OneVision 0.5B.

The official slices are:
- `RVS-Ego subsample5`
- `RVS-Ego subsample5_offset5`
- `RVS-Movie subsample5_movie`
- `RVS-Movie subsample5_movie_offset5`

## Main Claims Supported by the Current Results
1. `rekv` is the most consistently strong streaming baseline across the completed subsample slices.
2. `duo_streaming (s=0.5)` is a meaningful streaming Duo baseline against `full_streaming`, especially as a memory-saving alternative, but its quality-latency tradeoff is dataset-dependent.
3. `duo_plus_rekv (s=0.375)` is a technically valid hybrid and sometimes complementary to `rekv`, but it is not yet a universal improvement over plain `rekv`.

## Evidence Pattern
- On `RVS-Ego subsample5`, `duo_plus_rekv (s=0.375)` matches `rekv` on judge score and answer latency while using slightly less memory.
- On `RVS-Ego subsample5_offset5`, `duo_plus_rekv (s=0.375)` stays close to `rekv`, but still trails it on judge score and answer latency.
- On `RVS-Movie subsample5`, `rekv` remains the best operating point.
- On `RVS-Movie subsample5_offset5`, `duo_plus_rekv (s=0.375)` matches `rekv` on judge score and slightly lowers memory, but remains slower.

## What We Should Not Overclaim
- We should not claim that `duo_plus_rekv` already beats `rekv` consistently.
- We should not claim that `duo_streaming (s=0.5)` is always the best streaming operating point.
- We should not claim full-dataset validity yet, because full evaluation remains deferred.

## Recommended Framing
- `duo_streaming` vs `full_streaming`:
  - demonstrates how DuoAttention behaves as a streaming decoder-attention baseline
- `rekv`:
  - strongest standalone streaming memory baseline in the current subsample study
- `duo_plus_rekv`:
  - promising hybrid that preserves the correct ReKV retrieval semantics and sometimes complements `rekv`

## Deferred Work
- full-dataset evaluation
- any further A+B retuning
- broader significance claims beyond the current four subsample slices
