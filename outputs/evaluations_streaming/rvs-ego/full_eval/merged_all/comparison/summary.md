| dataset | slice_name | display_label | primary_quality_score | avg_answer_latency_sec | peak_memory_bytes | peak_cpu_offload_bytes | total_conversations_answered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rvs_ego | offset0 | full_streaming | 0.12080378161791668 | 1.770709854666916 | 16124705792 | None | 1465 |
| rvs_ego | offset0 | duo_streaming (s=0.75,sink=256,recent=512) | 0.13935540292036316 | 0.32658337034843704 | 5186595328 | None | 1465 |
| rvs_ego | offset0 | rekv (topk=64,n_local=15000) | 0.19262819927996733 | 0.4050044905455503 | 3360003072 | 4342431744 | 1465 |
| rvs_ego | offset0 | duo_plus_rekv (topk=64,s=0.75,sink=256,recent=512) | 0.19009763746816383 | 0.5899297900224953 | 3380327936 | 4337614848 | 1465 |
| rvs_ego | offset0 | streamingtom | 0.19857231755961582 | 0.9686122140465717 | 2417567232 | None | 1465 |
| rvs_ego | offset0 | duo_plus_streamingtom | 0.041677499947867826 | 2.3818251663501564 | 2463471104 | None | 1206 |
