
import argparse
import torch
import os
import tempfile
import json

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.utils import (
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)
from duo_attn.patch import enable_duo_attention_eval
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache
from duo_attn.loader import create_video_qa_dataloader
from duo_attn.train import build_llava_video_inputs_embeds

from duo_attn.eval.efficiency.utils import bench_func


def parse_benchmark_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LLaVA-OneVision prefill/decoding with DuoAttention"
    )
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model id or local path")
    
    parser.add_argument("--video_path", type=str, default=None, help="Path to a single video file")
    parser.add_argument("--prompt", type=str, default="Describe this video in detail.", help="Text prompt for single-video mode")
    
    parser.add_argument("--video_root", type=str, default=None, help="Root directory containing videos")
    parser.add_argument("--annotation_path", type=str, default=None, help="Path to annotation JSONL")
    
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--attn_load_dir", type=str, default=None, help="Path to attention pattern directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    
    parser.add_argument("--output_dir", type=str, default=None)

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
    from duo_attn.loader import VideoQADataset

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


if __name__ == "__main__":
    args = parse_benchmark_args()

    seed_everything(args.seed)
    print(f"Loading LLaVA-OneVision model: {args.model_name}")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=True
    )

    if args.attn_load_dir is not None:
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

    if args.video_path is not None:
        # Single-video mode
        print(f"Loading single video: {args.video_path}")
        batch = build_single_video_batch(
            args.video_path, args.prompt, processor, args.num_frames, args.max_length
        )
        
    elif args.video_root is not None and args.annotation_path is not None:
        # Dataset mode — grab the first sample
        print("Loading video from dataset...")
        dataloader = create_video_qa_dataloader(
            video_root=args.video_root,
            annotation_path=args.annotation_path,
            processor=processor,
            model_id=args.model_name,
            num_frames=args.num_frames,
            max_length=args.max_length,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        batch = next(iter(dataloader))
    else:
        raise ValueError(
            "Provide either --video_path for a single video, "
            "or both --video_root and --annotation_path for dataset mode."
        )

    batch = move_batch_to_device(batch, device, model_dtype)

    # Build multimodal inputs_embeds (vision + text)
    with torch.no_grad():
        inputs_embeds = build_llava_video_inputs_embeds(model, batch)

    seq_len = inputs_embeds.shape[1]
    print(f"Input sequence length (text + vision tokens): {seq_len}")

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

    # Run once more to capture outputs for decoding benchmark
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

    gen_latency, gen_memory = bench_func(decode_func, num_steps=100, num_warmup_steps=10)

    print("\n--- Results ---")
    print(f"Model: {args.model_name}")
    print(f"Num frames: {args.num_frames}")
    print(f"Input sequence length: {seq_len}")
    print(f"Sparsity: {sparsity}")
    print(f"Average prefill time: {ctx_latency:.2f} ms")
    print(f"Peak prefill memory: {ctx_memory:.2f} MB")
    print(f"Average generation time: {gen_latency:.2f} ms")
    print(f"Peak generation memory: {gen_memory:.2f} MB")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "benchmark_result.txt")
        with open(result_path, "w") as f:
            print(f"Average prefill time: {ctx_latency:.2f} ms", file=f)
            print(f"Peak prefill memory: {ctx_memory:.2f} MB", file=f)
            print(f"Average generation time: {gen_latency:.2f} ms", file=f)
            print(f"Peak generation memory: {gen_memory:.2f} MB", file=f)
            print(f"Model name: {args.model_name}", file=f)
            print(f"Num frames: {args.num_frames}", file=f)
            print(f"Input sequence length: {seq_len}", file=f)
            print(f"Sparsity: {sparsity}", file=f)
        print(f"Results saved to {result_path}")
