| dataset | slice_name | display_label | primary_quality_score | avg_answer_latency_sec | peak_memory_bytes | peak_cpu_offload_bytes | total_conversations_answered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rvs_movie | offset0 | full_streaming | 0.7670341207348836 | 1.3400275094591674 | 9877919744 | None | 1905 |
| rvs_movie | offset0 | duo_streaming (s=0.75,sink=256,recent=512) | 0.7771128608923633 | 0.3325039373570809 | 3726841344 | None | 1905 |
| rvs_movie | offset0 | rekv (topk=64,n_local=15000) | 0.7636745406823899 | 0.6341371222832143 | 3414142976 | 2444574720 | 1905 |
| rvs_movie | offset0 | duo_plus_rekv (topk=64,s=0.75,sink=256,recent=512) | 0.7677690288713661 | 0.8907937005978063 | 3421976064 | 2439757824 | 1905 |
