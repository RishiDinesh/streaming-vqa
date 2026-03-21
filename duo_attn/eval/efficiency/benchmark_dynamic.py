import os
from typing import Any, Dict

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.patch import enable_duo_attention_eval
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
    prefill_kwargs: Dict[str, Any] = {}

    if is_llava_onevision_model(args.model_name):
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

        if args.attn_load_dir is not None:
            full_attention_heads, sink_size, recent_size = load_attn_pattern(
                args.attn_load_dir
            )
            full_attention_heads, sparsity = sparsify_attention_heads(
                full_attention_heads, args.threshold, args.sparsity
            )
            print(
                "Applying DuoAttention eval patch for LLaVA-OneVision "
                f"(sink={sink_size}, recent={recent_size}, sparsity={sparsity})"
            )
            enable_duo_attention_eval(
                model,
                full_attention_heads,
                sink_size,
                recent_size,
            )

        model_dtype = next(model.parameters()).dtype
        model_inputs, mm_diagnostics = build_llava_first_sample_inputs(
            args,
            processor,
            device=device,
            model_dtype=model_dtype,
        )
        print("Using first Video-QA sample for dynamic multimodal benchmark.")
        print("Multimodal diagnostics:")
        for key in sorted(mm_diagnostics):
            print(f"{key}: {mm_diagnostics[key]}")

        prefill_kwargs = dict(model_inputs)
    else:
        tokenizer = get_tokenizer(args.model_name)
        with torch.no_grad():
            model = get_model(args.model_name)

        if model.config.model_type == "mistral":
            model.model._prepare_decoder_attention_mask = lambda *a, **k: None
        elif model.config.model_type == "llama":
            model.model._prepare_decoder_attention_mask = lambda *a, **k: None

        model = to_device(model, args.device)
        model_device = next(model.parameters()).device

        if args.attn_load_dir is not None:
            full_attention_heads, sink_size, recent_size = load_attn_pattern(
                args.attn_load_dir
            )
            full_attention_heads, sparsity = sparsify_attention_heads(
                full_attention_heads, args.threshold, args.sparsity
            )
            enable_duo_attention_eval(
                model,
                full_attention_heads,
                sink_size,
                recent_size,
            )

        text = "a\n\n" * args.max_length
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model_device)[
            :, : args.max_length - 1
        ]
        print(input_ids.shape)
        prefill_kwargs = {"input_ids": input_ids}

    def func1():
        with torch.no_grad():
            _ = model(
                **prefill_kwargs,
                past_key_values=None,
                use_cache=True,
            )

    ctx_latency, ctx_memory = bench_func(func1, num_steps=10, num_warmup_steps=3)

    with torch.no_grad():
        outputs = model(
            **prefill_kwargs,
            past_key_values=None,
            use_cache=True,
        )

    print(
        "Peak memory usage in the pre-filling stage: "
        f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB"
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    def func2():
        with torch.no_grad():
            _ = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )

    gen_latency, gen_memory = bench_func(func2, num_steps=100, num_warmup_steps=10)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "benchmark_result.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            print(f"Average generation time: {gen_latency:.2f} ms", file=f)
            print(f"Peak generation memory usage: {gen_memory:.2f} MB", file=f)
            print(f"Average context time: {ctx_latency:.2f} ms", file=f)
            print(f"Peak context memory usage: {ctx_memory:.2f} MB", file=f)
            print(f"Model name: {args.model_name}", file=f)
            print(f"Context length: {args.max_length}", file=f)
            print(f"Sparsity: {sparsity}", file=f)
            if mm_diagnostics:
                print("Multimodal diagnostics:", file=f)
                for key in sorted(mm_diagnostics):
                    print(f"{key}: {mm_diagnostics[key]}", file=f)
