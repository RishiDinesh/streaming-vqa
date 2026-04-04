# Streaming Evaluation Outputs

This directory contains both curated final outputs and exploratory/intermediate artifacts for the streaming `ReKV` vs `DuoAttention` work.

Promoted final package:
- `final_subsample_package/`

Promoted summaries outside the final package:
- `rvs-ego/subsample_comparison_offset0_vs_offset5/`
- `rvs-movie/subsample_comparison_offset0_vs_offset5/`
- `subsample_only_summary/`

Exploratory or intermediate material:
- smoke runs such as `ab_smoke1/` and `movie_smoke1/`
- A+B tuning directories such as `subsample5_ab_tuning/`, `subsample5_offset5_ab_tuning/`, and `*_recent1536/`
- slice-local debug plot directories outside the curated final package

When in doubt, start with:
- `final_subsample_package/README.md`
- `final_subsample_package/final_metrics.md`
- `final_subsample_package/paper_story.md`
- `final_subsample_package/qualitative_examples.md`
