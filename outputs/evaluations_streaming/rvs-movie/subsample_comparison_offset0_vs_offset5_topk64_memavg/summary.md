| dataset | slice_name | display_label | primary_quality_score | avg_answer_latency_sec | peak_memory_bytes | peak_cpu_offload_bytes | total_conversations_answered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rvs_movie | subsample5_movie | full_streaming | 0.7733333333333335 | 0.5146111484000964 | 2997906432 | None | 15 |
| rvs_movie | subsample5_movie | duo_streaming (s=0.5) | 0.7733333333333334 | 0.5029197842658808 | 3062294016 | None | 15 |
| rvs_movie | subsample5_movie | rekv (topk=64,n_local=15000) | 0.7733333333333334 | 0.6775483731997762 | 3073405440 | 744210432 | 15 |
| rvs_movie | subsample5_movie | duo_plus_rekv (topk=64,s=0.5) | 0.7466666666666667 | 0.9062557553998583 | 3078333952 | 739393536 | 15 |
| rvs_movie | subsample5_movie | duo_plus_rekv (topk=64,s=0.75) | 0.7733333333333334 | 0.8523471346670703 | 3071766016 | 739393536 | 15 |
| rvs_movie | subsample5_movie_offset5 | full_streaming | 0.7466666666666667 | 0.3389860619330041 | 4156259328 | None | 15 |
| rvs_movie | subsample5_movie_offset5 | duo_streaming (s=0.5) | 0.7733333333333335 | 0.47063207626658066 | 4279611392 | None | 15 |
| rvs_movie | subsample5_movie_offset5 | rekv (topk=64,n_local=15000) | 0.7866666666666668 | 0.7149072100665459 | 2964370432 | 1536589824 | 15 |
| rvs_movie | subsample5_movie_offset5 | duo_plus_rekv (topk=64,s=0.5) | 0.8000000000000002 | 1.107148526933209 | 2974434816 | 1531772928 | 15 |
| rvs_movie | subsample5_movie_offset5 | duo_plus_rekv (topk=64,s=0.75) | 0.8000000000000002 | 1.0564303070680277 | 2967933952 | 1531772928 | 15 |
