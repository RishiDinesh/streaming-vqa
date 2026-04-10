| dataset | slice_name | display_label | primary_quality_score | avg_answer_latency_sec | peak_memory_bytes | peak_cpu_offload_bytes | total_conversations_answered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rvs_movie | subsample5_movie | full_streaming | 0.7733333333333335 | 0.4889694402668586 | 2998404608 | None | 15 |
| rvs_movie | subsample5_movie | duo_streaming (s=0.5) | 0.7733333333333334 | 0.5032399087339097 | 3062294016 | None | 15 |
| rvs_movie | subsample5_movie | rekv (topk=32,n_local=15000) | 0.7866666666666667 | 0.5928763095995236 | 2822692864 | 744210432 | 15 |
| rvs_movie | subsample5_movie | duo_plus_rekv (topk=32,s=0.5) | 0.7866666666666667 | 0.8483253905333792 | 2831975936 | 739393536 | 15 |
| rvs_movie | subsample5_movie | duo_plus_rekv (topk=32,s=0.75) | 0.7866666666666668 | 0.920597787466492 | 2828547584 | 739393536 | 15 |
| rvs_movie | subsample5_movie_offset5 | full_streaming | 0.7466666666666667 | 0.34148857633311613 | 4156259328 | None | 15 |
| rvs_movie | subsample5_movie_offset5 | duo_streaming (s=0.5) | 0.7733333333333335 | 0.4684996080672136 | 4279611392 | None | 15 |
| rvs_movie | subsample5_movie_offset5 | rekv (topk=32,n_local=15000) | 0.8000000000000002 | 0.6223867689999073 | 2715883008 | 1536589824 | 15 |
| rvs_movie | subsample5_movie_offset5 | duo_plus_rekv (topk=32,s=0.5) | 0.8000000000000002 | 0.7966785297993435 | 2726424064 | 1531772928 | 15 |
| rvs_movie | subsample5_movie_offset5 | duo_plus_rekv (topk=32,s=0.75) | 0.8000000000000002 | 0.7470027480000377 | 2722996736 | 1531772928 | 15 |
