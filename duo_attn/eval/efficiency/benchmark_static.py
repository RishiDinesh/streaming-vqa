import os
from typing import Any, Dict

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.patch.llama import enable_llama_duo_attention_static_kv_cache_eval
from duo_attn.patch.llava_onevision import (
    enable_llava_onevision_duo_attention_static_kv_cache_eval,
)
from duo_attn.patch.static_kv_cache import DuoAttentionStaticKVCache
from duo_attn.utils import (
    get_model,
    get_tokenizer,
    load_attn_pattern,
    parse_args,
    seed_everything,
    sparsify_attention_heads,
    to_device,
)
from utils import (
    bench_func,
    build_llava_first_sample_inputs,
    is_llava_onevision_model,
    resolve_cuda_device,
)


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    sparsity = None
    mm_diagnostics: Dict[str, Any] = {}
    prefill_mode = "chunked_text"

    if is_llava_onevision_model(args.model_name):
        if args.attn_load_dir is None:
            raise ValueError(
                "LLaVA-OneVision static benchmark requires --attn_load_dir."
            )

        device = resolve_cuda_device(args.device)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
        with torch.no_grad():
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                args.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
        model.to(device)
        model.eval()

        full_attention_heads, sink_size, recent_size = load_attn_pattern(args.attn_load_dir)
        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, args.threshold, args.sparsity
        )
        print(f"True Sparsity: {sparsity}")
        enable_llava_onevision_duo_attention_static_kv_cache_eval(
            model, full_attention_heads
        )

        model_dtype = next(model.parameters()).dtype
        model_inputs, mm_diagnostics = build_llava_first_sample_inputs(
            args,
            processor,
            device=device,
            model_dtype=model_dtype,
        )
        max_size = model_inputs["input_ids"].shape[1] + 5
        prefilling_chunk_size = args.prefilling_chunk_size
        prefill_mode = "chunked_multimodal"
        print(
            "Using first Video-QA sample for static multimodal benchmark "
            f"(prompt tokens={mm_diagnostics['prompt_token_length']})."
        )
        print(f"Max size: {max_size}, Prefilling chunk size: {prefilling_chunk_size}")
        print("Multimodal diagnostics:")
        for key in sorted(mm_diagnostics):
            print(f"{key}: {mm_diagnostics[key]}")

        kv_cache = DuoAttentionStaticKVCache(
            model,
            full_attention_heads,
            1,
            max_size,
            sink_size,
            recent_size,
        )

        multimodal_static_keys = [
            key for key in model_inputs.keys() if key not in {"input_ids", "attention_mask"}
        ]

        def build_mm_chunk_inputs(start: int, end: int, include_multimodal: bool):
            chunk_inputs = {
                "input_ids": model_inputs["input_ids"][:, start:end],
            }
            if "attention_mask" in model_inputs:
                chunk_inputs["attention_mask"] = model_inputs["attention_mask"][:, start:end]
            if include_multimodal:
                for key in multimodal_static_keys:
                    chunk_inputs[key] = model_inputs[key]
            return chunk_inputs

        def func1():
            kv_cache.clear()
            with torch.no_grad():
                seq_len = model_inputs["input_ids"].shape[1]
                for i in range(0, seq_len, prefilling_chunk_size):
                    input_chunk = build_mm_chunk_inputs(
                        i,
                        min(i + prefilling_chunk_size, seq_len),
                        include_multimodal=(i == 0),
                    )
                    _ = model(
                        **input_chunk,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
            kv_cache.clear()

        ctx_latency, ctx_memory = bench_func(func1, num_steps=10, num_warmup_steps=3)

        kv_cache.clear()
        with torch.no_grad():
            seq_len = model_inputs["input_ids"].shape[1]
            for i in range(0, seq_len, prefilling_chunk_size):
                input_chunk = build_mm_chunk_inputs(
                    i,
                    min(i + prefilling_chunk_size, seq_len),
                    include_multimodal=(i == 0),
                )
                outputs = model(
                    **input_chunk,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
    else:
        if args.attn_load_dir is None:
            raise ValueError(
                "Static benchmark requires --attn_load_dir for DuoAttention static KV cache."
            )

        tokenizer = get_tokenizer(args.model_name)
        with torch.no_grad():
            model = get_model(args.model_name)
        model.eval()
        model = to_device(model, args.device)
        model_device = next(model.parameters()).device

        full_attention_heads, sink_size, recent_size = load_attn_pattern(args.attn_load_dir)
        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, args.threshold, args.sparsity
        )
        print(f"True Sparsity: {sparsity}")
        enable_llama_duo_attention_static_kv_cache_eval(model, full_attention_heads)

        text = "a\n\n" * args.max_length
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model_device)[
            :, : args.max_length - 1
        ]
        print(input_ids.shape)
        max_size = input_ids.size(1) + 5
        prefilling_chunk_size = args.prefilling_chunk_size
        print(f"Max size: {max_size}, Prefilling chunk size: {prefilling_chunk_size}")

        kv_cache = DuoAttentionStaticKVCache(
            model,
            full_attention_heads,
            1,
            max_size,
            sink_size,
            recent_size,
        )

        def func1():
            with torch.no_grad():
                for i in range(0, input_ids.size(1), prefilling_chunk_size):
                    input_chunk = input_ids[:, i : i + prefilling_chunk_size]
                    _ = model(
                        input_ids=input_chunk,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
                kv_cache.clear()

        ctx_latency, ctx_memory = bench_func(func1, num_steps=10, num_warmup_steps=3)

        kv_cache.clear()
        with torch.no_grad():
            for i in range(0, input_ids.size(1), prefilling_chunk_size):
                input_chunk = input_ids[:, i : i + prefilling_chunk_size]
                outputs = model(
                    input_ids=input_chunk,
                    past_key_values=kv_cache,
                    use_cache=True,
                )

    print(
        "Peak memory usage in the pre-filling stage: "
        f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB"
    )
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    def func2():
        with torch.no_grad():
            _ = model(
                input_ids=pred_token_idx,
                past_key_values=kv_cache,
                use_cache=True,
            )
        kv_cache.evict_last(1)

    gen_latency, gen_memory = bench_func(func2, num_steps=100, num_warmup_steps=50)

    kv_cache_memory_usage = kv_cache.memory_usage / 1024 / 1024
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "benchmark_result.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            print(f"Average generation time: {gen_latency:.4f} ms", file=f)
            print(f"Peak generation memory usage: {gen_memory:.4f} MB", file=f)
            print(f"Average context time: {ctx_latency:.4f} ms", file=f)
            print(f"Peak context memory usage: {ctx_memory:.4f} MB", file=f)
            print(f"Model name: {args.model_name}", file=f)
            print(f"Context length: {args.max_length}", file=f)
            print(f"Sparsity: {sparsity}", file=f)
            print(f"Prefill mode: {prefill_mode}", file=f)
            if prefill_mode in {"chunked_text", "chunked_multimodal"}:
                print(f"Prefilling chunk size: {args.prefilling_chunk_size}", file=f)
            print(f"KV cache memory usage: {kv_cache_memory_usage:.4f} MB", file=f)
            if mm_diagnostics:
                print("Multimodal diagnostics:", file=f)
                for key in sorted(mm_diagnostics):
                    print(f"{key}: {mm_diagnostics[key]}", file=f)
