import argparse
import json
import os
import tempfile

import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.eval.efficiency.utils import bench_func
from duo_attn.data.loader import create_video_qa_dataloader
from duo_attn.patch import enable_duo_attention_eval
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache
from duo_attn.train import build_llava_video_inputs_embeds
from duo_attn.utils import (
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)


def parse_benchmark_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LLaVA-OneVision prefill/decoding with DuoAttention"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model id or local path",
    )

    parser.add_argument(
        "--video_path", type=str, default=None, help="Path to a single video file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this video in detail.",
        help="Text prompt for single-video mode",
    )

    parser.add_argument(
        "--video_root", type=str, default=None, help="Root directory containing videos"
    )
    parser.add_argument(
        "--annotation_path", type=str, default=None, help="Path to annotation JSONL"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["video_qa", "egoschema"],
        default="video_qa",
        help="Dataset loader type to use when --video_root/--annotation_path are provided.",
    )
    parser.add_argument(
        "--egoschema_default_video_ext",
        type=str,
        default=".mp4",
        help="Default extension for EgoSchema entries that store video ids without suffix.",
    )
    parser.add_argument(
        "--egoschema_include_options_in_question",
        action="store_true",
        help="For EgoSchema, append answer choices into the question prompt.",
    )

    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dataset-mode dataloader.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention_mode",
        type=str,
        choices=["auto", "duo", "baseline"],
        default="auto",
        help=(
            "'duo' enables DuoAttention using --attn_load_dir, "
            "'baseline' runs without DuoAttention, and "
            "'auto' uses duo only when --attn_load_dir is provided."
        ),
    )

    parser.add_argument(
        "--attn_load_dir",
        type=str,
        default=None,
        help="Path to attention pattern directory",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--ui_style",
        type=str,
        choices=["benchmark", "demo"],
        default="benchmark",
        help="Terminal UI style. 'benchmark' keeps tqdm warmup/benchmark bars; 'demo' shows stage-like bars.",
    )
    parser.add_argument(
        "--prefill_chunk_size",
        type=int,
        default=32000,
        help="Chunk size for demo prefill UI mode.",
    )
    parser.add_argument(
        "--decode_tokens",
        type=int,
        default=100,
        help="Number of decode steps in demo UI mode.",
    )

    return parser.parse_args()


def move_batch_to_device(batch, device, model_dtype):
    """Move a batch dict to the target device, casting floats to model_dtype."""
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=model_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def build_single_video_batch(video_path, prompt, processor, num_frames, max_length):
    """
    Build a single-sample batch from one video file, without needing
    an annotation file or video root directory.
    """
    from duo_attn.data.vnbench import VideoQADataset

    # Create a temporary annotation file pointing to the single video
    annotation = {
        "video": os.path.abspath(video_path),
        "question": prompt,
        "gt": "N/A",  # dummy ground truth (not used for benchmarking)
    }
    tmp_annotation = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    )
    json.dump(annotation, tmp_annotation)
    tmp_annotation.write("\n")
    tmp_annotation.close()

    try:
        dataset = VideoQADataset(
            video_root=os.path.dirname(os.path.abspath(video_path)),
            annotation_path=tmp_annotation.name,
            processor=processor,
            num_frames=num_frames,
            max_length=max_length,
        )
        sample = dataset[0]
        # Collate into a batch of size 1 — add batch dimension
        batch = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                batch[key] = value.unsqueeze(0)
            else:
                batch[key] = value
        return batch
    finally:
        os.unlink(tmp_annotation.name)


def run_benchmark_mode(model, inputs_embeds):
    print("\n--- Pre-filling benchmark ---")
    torch.cuda.reset_peak_memory_stats()

    def prefill_func():
        with torch.no_grad():
            _ = model.language_model(
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                use_cache=True,
            )

    ctx_latency, ctx_memory = bench_func(prefill_func, num_steps=10, num_warmup_steps=3)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
        )
    prefill_peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory usage in the pre-filling stage: {prefill_peak_memory:.2f} MB")

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    print("\n--- Decoding benchmark ---")

    def decode_func():
        with torch.no_grad():
            _ = model.language_model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )

    gen_latency, gen_memory = bench_func(
        decode_func, num_steps=100, num_warmup_steps=10
    )

    return {
        "ctx_latency": ctx_latency,
        "ctx_memory": ctx_memory,
        "gen_latency": gen_latency,
        "gen_memory": gen_memory,
        "prefill_total_ms": None,
    }


def run_demo_mode(model, inputs_embeds, decode_tokens, prefill_chunk_size):
    seq_len = inputs_embeds.shape[1]
    chunk_size = max(1, int(prefill_chunk_size))
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    print("\n--- Pre-filling stage ---")
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    past_key_values = None
    pred_token_idx = None
    start.record()
    pbar = tqdm(range(num_chunks))
    for i in pbar:
        s = i * chunk_size
        e = min((i + 1) * chunk_size, seq_len)
        chunk = inputs_embeds[:, s:e, :]
        with torch.no_grad():
            outputs = model.language_model(
                inputs_embeds=chunk,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        mem_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        pbar.set_description(f"Pre-filling ({e}/{seq_len}, Mem: {mem_gb:.2f} GB)")
    end.record()
    torch.cuda.synchronize()

    prefill_total_ms = start.elapsed_time(end)
    ctx_latency = prefill_total_ms / max(1, num_chunks)
    ctx_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Pre-filling time: {prefill_total_ms / 1000:.2f}s")

    print("\n--- Decoding stage ---")
    torch.cuda.reset_peak_memory_stats()
    decode_total_ms = 0.0
    decode_steps = max(1, int(decode_tokens))
    pbar = tqdm(range(decode_steps))
    for step_idx, _ in enumerate(pbar, start=1):
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record()
        with torch.no_grad():
            outputs = model.language_model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
        step_end.record()
        torch.cuda.synchronize()
        step_ms = step_start.elapsed_time(step_end)
        decode_total_ms += step_ms

        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        avg_ms = decode_total_ms / step_idx
        mem_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        pbar.set_description(
            f"Decoding (Mem: {mem_gb:.2f} GB | Latency: {avg_ms:.2f} ms/tok)"
        )

    gen_latency = decode_total_ms / decode_steps
    gen_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"Per-token decoding latency: {gen_latency:.2f} ms")
    print(f"Peak memory: {gen_memory / 1024:.2f} GB")

    return {
        "ctx_latency": ctx_latency,
        "ctx_memory": ctx_memory,
        "gen_latency": gen_latency,
        "gen_memory": gen_memory,
        "prefill_total_ms": prefill_total_ms,
    }


if __name__ == "__main__":
    args = parse_benchmark_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")

    seed_everything(args.seed)
    print(f"Loading LLaVA-OneVision model: {args.model_name}")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    attention_mode = args.attention_mode
    if attention_mode == "auto":
        attention_mode = "duo" if args.attn_load_dir is not None else "baseline"
    print(f"Attention mode: {attention_mode}")

    if attention_mode == "duo":
        if args.attn_load_dir is None:
            raise ValueError(
                "--attn_load_dir is required when --attention_mode=duo."
            )
        full_attention_heads, sink_size, recent_size = load_attn_pattern(
            args.attn_load_dir
        )

        if args.sink_size is not None:
            sink_size = args.sink_size
        if args.recent_size is not None:
            recent_size = args.recent_size

        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, args.threshold, args.sparsity
        )
        print(f"True sparsity: {sparsity}")

        enable_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    else:
        sparsity = 0.0
        enable_tuple_kv_cache(model)

    # Move model to GPU
    device = torch.device("cuda")
    model_dtype = torch.bfloat16
    model = model.to(device)



    if args.dataset_type == "egoschema" and args.video_path is None:
        if args.video_root is None and os.path.isdir("data/videos"):
            args.video_root = "data/videos"
            print(f"Using default EgoSchema video_root: {args.video_root}")
        if args.annotation_path is None and os.path.isfile("data/questions.json"):
            args.annotation_path = "data/questions.json"
            print(f"Using default EgoSchema annotation_path: {args.annotation_path}")

    dataloader = None
    if args.video_path is not None:
        print(f"Loading single video: {args.video_path}")
        single_batch = build_single_video_batch(
            args.video_path, args.prompt, processor, args.num_frames, args.max_length
        )
        total_batches = 1
        batch_iter = [(1, single_batch)]
    elif args.video_root is not None and args.annotation_path is not None:
        print("Loading video dataset...")
        dataset_name = "egoschema" if args.dataset_type == "egoschema" else "vnbench"
        dataloader = create_video_qa_dataloader(
            video_root=args.video_root,
            dataset_name=dataset_name,
            annotation_path=args.annotation_path,
            processor=processor,
            model_id=args.model_name,
            num_frames=args.num_frames,
            max_length=args.max_length,
            include_options_in_question=args.egoschema_include_options_in_question,
            default_video_ext=args.egoschema_default_video_ext,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        total_batches = len(dataloader)
        batch_iter = enumerate(dataloader, start=1)
    else:
        raise ValueError(
            "Provide either --video_path for a single video, "
            "or both --video_root and --annotation_path for dataset mode."
        )

    ctx_latency_sum = 0.0
    gen_latency_sum = 0.0
    ctx_memory_peak = 0.0
    gen_memory_peak = 0.0
    seq_len_sum = 0
    prefill_total_ms_sum = 0.0
    prefill_total_ms_count = 0

    for batch_idx, batch in batch_iter:
        print(f"[Batch {batch_idx}/{total_batches}] Starting benchmark...")

        batch = move_batch_to_device(batch, device, model_dtype)

        with torch.no_grad():
            inputs_embeds = build_llava_video_inputs_embeds(model, batch)

        seq_len = inputs_embeds.shape[1]
        seq_len_sum += int(seq_len)
        print(f"[Batch {batch_idx}/{total_batches}] Input sequence length: {seq_len}")

        if args.ui_style == "demo":
            result = run_demo_mode(
                model,
                inputs_embeds,
                decode_tokens=args.decode_tokens,
                prefill_chunk_size=args.prefill_chunk_size,
            )
        else:
            result = run_benchmark_mode(model, inputs_embeds)

        ctx_latency_sum += float(result["ctx_latency"])
        gen_latency_sum += float(result["gen_latency"])
        ctx_memory_peak = max(ctx_memory_peak, float(result["ctx_memory"]))
        gen_memory_peak = max(gen_memory_peak, float(result["gen_memory"]))

        prefill_total_ms = result.get("prefill_total_ms", None)
        if prefill_total_ms is not None:
            prefill_total_ms_sum += float(prefill_total_ms)
            prefill_total_ms_count += 1

    num_batches = max(1, total_batches)
    avg_seq_len = seq_len_sum / num_batches
    ctx_latency = ctx_latency_sum / num_batches
    gen_latency = gen_latency_sum / num_batches
    ctx_memory = ctx_memory_peak
    gen_memory = gen_memory_peak
    prefill_total_ms = (
        prefill_total_ms_sum / prefill_total_ms_count
        if prefill_total_ms_count > 0
        else None
    )

    print("\n--- Results ---")
    print(f"Model: {args.model_name}")
    print(f"Num frames: {args.num_frames}")
    print(f"Processed batches: {total_batches}")
    print(f"Average input sequence length: {avg_seq_len:.2f}")
    print(f"Sparsity: {sparsity}")
    if prefill_total_ms is not None:
        print(f"Average pre-filling time: {prefill_total_ms / 1000:.2f} s")
    print(f"Average prefill latency: {ctx_latency:.2f} ms")
    print(f"Peak prefill memory across batches: {ctx_memory:.2f} MB")
    print(f"Average generation latency: {gen_latency:.2f} ms")
    print(f"Peak generation memory across batches: {gen_memory:.2f} MB")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "benchmark_result.txt")
        with open(result_path, "w") as f:
            if prefill_total_ms is not None:
                print(f"Average pre-filling time: {prefill_total_ms / 1000:.2f} s", file=f)
            print(f"Average prefill latency: {ctx_latency:.2f} ms", file=f)
            print(f"Peak prefill memory across batches: {ctx_memory:.2f} MB", file=f)
            print(f"Average generation latency: {gen_latency:.2f} ms", file=f)
            print(f"Peak generation memory across batches: {gen_memory:.2f} MB", file=f)
            print(f"Model name: {args.model_name}", file=f)
            print(f"Num frames: {args.num_frames}", file=f)
            print(f"Processed batches: {total_batches}", file=f)
            print(f"Average input sequence length: {avg_seq_len:.2f}", file=f)
            print(f"Sparsity: {sparsity}", file=f)
        print(f"Results saved to {result_path}")
