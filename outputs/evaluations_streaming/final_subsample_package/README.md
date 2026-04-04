# Final Subsample Package

This directory is the curated, publication-oriented output bundle for the streaming `ReKV` vs `DuoAttention` work on LLaVA-OneVision 0.5B.

It promotes only the validated subsample-only results:
- `RVS-Ego subsample5`
- `RVS-Ego subsample5_offset5`
- `RVS-Movie subsample5_movie`
- `RVS-Movie subsample5_movie_offset5`

Official methods in this package:
- `full_streaming`
- `duo_streaming (s=0.5)`
- `duo_streaming (s=0.0)` as a control
- `rekv`
- `duo_plus_rekv (s=0.375)` as the current best hybrid

Package layout:
- `rvs-ego/`
  - promoted per-slice result JSONs
  - promoted per-slice plot bundles
  - promoted cross-slice comparison bundle
- `rvs-movie/`
  - promoted per-slice result JSONs
  - promoted per-slice plot bundles
  - promoted cross-slice comparison bundle
- `global/`
  - cross-dataset subsample-only summary bundle
- `final_metrics.csv`
  - compact metrics table with judge score, ROUGE-L F1, token F1, answer latency, and peak memory
- `final_metrics.md`
  - Markdown view of the same metrics table
- `promoted_runs.json`
  - machine-readable manifest of the promoted runs
- `paper_story.md`
  - compact narrative of the current research story
- `qualitative_examples.md`
  - curated comparison examples for interpretation
- `qualitative_examples.json`
  - machine-readable version of the curated examples

What is intentionally not promoted here:
- smoke-only outputs
- exploratory A+B tuning runs other than the promoted `duo_plus_rekv (s=0.375)` setting
- debug-only intermediate files that are not needed for the main story

Those exploratory artifacts still exist elsewhere under `outputs/evaluations_streaming/...`, but this directory is the stable bundle to cite, inspect, and push.
