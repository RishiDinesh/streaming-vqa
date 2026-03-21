import argparse
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.data import create_video_qa_dataloader
from duo_attn.patch import enable_duo_attention_eval
from duo_attn.utils import load_attn_pattern, seed_everything, sparsify_attention_heads

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference import (  # noqa: E402
    build_generation_prompt_inputs,
    build_llava_video_inputs_embeds,
    canonicalize_secret_word,
    extract_secret_word_candidate,
    greedy_generate_from_prompt,
    prepare_model_inputs,
)
from runtime import resolve_device_and_dtype  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-sample VNBench inference debug harness."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="./datasets/vnbench/videos",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="./datasets/vnbench/anno.jsonl",
    )
    parser.add_argument("--num_frames", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=32000)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--video_answer_prefix",
        type=str,
        default="",
        help=(
            "Optional answer prefix appended to the generation prompt before "
            "decoding. Leave empty to let the model generate the full answer."
        ),
    )
    parser.add_argument("--disable_video_chat_template", action="store_true")
    parser.add_argument("--attn_load_dir", type=str, default="./outputs/0p5b_sink512_recent1024_maxlen32000_frames128_depth0p1-0p8_needles5")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto|cpu|cuda[:index]",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--compare_generate",
        action="store_true",
        default=True,
        help="Also run model.generate() on the same prompt for comparison.",
    )
    parser.add_argument(
        "--no_compare_generate",
        dest="compare_generate",
        action="store_false",
    )
    return parser.parse_args()


def ensure_generation_pad_token(model, processor) -> None:
    if model.generation_config.pad_token_id is None:
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            eos_token_id = model.generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                pad_token_id = eos_token_id[0] if eos_token_id else None
            else:
                pad_token_id = eos_token_id
        if pad_token_id is not None:
            model.generation_config.pad_token_id = pad_token_id


def decode_ids(tokenizer, ids: List[int], skip_special_tokens: bool) -> str:
    return tokenizer.decode(
        ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=True,
    )


def token_strings(tokenizer, ids: List[int]) -> List[str]:
    return [tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in ids]


def is_immediate_im_end_only(tokenizer, ids: List[int]) -> bool:
    if not ids:
        return False

    non_pad_ids = [
        int(token_id)
        for token_id in ids
        if tokenizer.pad_token_id is None or int(token_id) != tokenizer.pad_token_id
    ]
    if not non_pad_ids:
        return False

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id < 0:
        return False
    return len(non_pad_ids) == 1 and non_pad_ids[0] == int(im_end_id)


def print_block(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device, dtype = resolve_device_and_dtype(args)

    print_block("Runtime")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"compare_generate={args.compare_generate}")
    print(f"video_answer_prefix={repr(args.video_answer_prefix)}")

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if "llava_onevision" not in config.model_type:
        raise ValueError(
            f"Expected a Llava-OneVision model, got model_type={config.model_type}."
        )

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    ensure_generation_pad_token(model, processor)

    if args.attn_load_dir:
        full_attention_heads, sink_size, recent_size = load_attn_pattern(
            args.attn_load_dir
        )
        full_attention_heads, actual_sparsity = sparsify_attention_heads(
            full_attention_heads,
            threshold=args.threshold,
            sparsity=args.sparsity,
        )
        print_block("DuoAttention")
        print(f"attn_load_dir={args.attn_load_dir}")
        print(f"sink_size={sink_size}")
        print(f"recent_size={recent_size}")
        print(f"actual_sparsity={actual_sparsity:.4f}")
        enable_duo_attention_eval(model, full_attention_heads, sink_size, recent_size)

    dataloader = create_video_qa_dataloader(
        video_root=args.video_root,
        dataset_name="vnbench",
        annotation_path=args.annotation_path,
        processor=processor,
        model_id=args.model_name,
        num_frames=args.num_frames,
        max_length=args.max_length,
        use_chat_template=not args.disable_video_chat_template,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    dataset = dataloader.dataset
    batch = next(iter(dataloader))
    labels = batch["labels"].to(device=device)
    model_inputs = prepare_model_inputs(batch, device, dtype)
    prompt_inputs = build_generation_prompt_inputs(
        model_inputs=model_inputs,
        labels=labels,
        tokenizer=processor.tokenizer,
        answer_prefix=args.video_answer_prefix,
    )

    prompt_input_ids = prompt_inputs["input_ids"][0].detach().cpu().tolist()
    prompt_text_skip = decode_ids(processor.tokenizer, prompt_input_ids, True)
    prompt_text_raw = decode_ids(processor.tokenizer, prompt_input_ids, False)

    raw_sample: Dict[str, Any] = {}
    if hasattr(dataset, "samples") and dataset.samples:
        raw_sample = dict(dataset.samples[0])

    print_block("Sample")
    print(f"sample_index=0")
    print(f"question={raw_sample.get('question', '')}")
    print(f"gt={raw_sample.get('gt', '')}")
    print(f"video={raw_sample.get('video', '')}")
    print(f"options={raw_sample.get('options', [])}")
    print(f"canonical_gt={canonicalize_secret_word(str(raw_sample.get('gt', '')))}")

    print_block("Prompt")
    print(f"prompt_input_len={len(prompt_input_ids)}")
    print(f"prompt_decode_skip_special_tokens=True:\n{repr(prompt_text_skip)}")
    print(f"prompt_decode_skip_special_tokens=False:\n{repr(prompt_text_raw)}")

    generated_ids = greedy_generate_from_prompt(
        model=model,
        prompt_inputs=prompt_inputs,
        max_new_tokens=args.max_new_tokens,
        tokenizer=processor.tokenizer,
    )[0].detach().cpu().tolist()

    generated_text_skip = decode_ids(processor.tokenizer, generated_ids, True).strip()
    generated_text_raw = decode_ids(processor.tokenizer, generated_ids, False).strip()
    generated_tokens = token_strings(processor.tokenizer, generated_ids)
    candidate = extract_secret_word_candidate(generated_text_skip)

    print_block("Manual Greedy Decode")
    print(f"generated_token_ids={generated_ids}")
    print(f"generated_tokens={generated_tokens}")
    print(f"generated_text_skip_special_tokens=True:\n{repr(generated_text_skip)}")
    print(f"generated_text_skip_special_tokens=False:\n{repr(generated_text_raw)}")
    print(f"secret_word_candidate={repr(candidate)}")
    print(f"is_immediate_im_end_only={is_immediate_im_end_only(processor.tokenizer, generated_ids)}")

    if generated_ids:
        print(f"starts_with_1={str(generated_text_skip).lstrip().startswith('1')}")

    if args.compare_generate:
        with torch.inference_mode():
            generated_full = model.generate(
                **prompt_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        prompt_len = prompt_inputs["input_ids"].shape[1]
        generated_full_ids = generated_full[0, prompt_len:].detach().cpu().tolist()
        generated_full_text_skip = decode_ids(
            processor.tokenizer, generated_full_ids, True
        ).strip()
        generated_full_text_raw = decode_ids(
            processor.tokenizer, generated_full_ids, False
        ).strip()

        print_block("model.generate Compare")
        print(f"generated_token_ids={generated_full_ids}")
        print(f"generated_tokens={token_strings(processor.tokenizer, generated_full_ids)}")
        print(
            "generated_text_skip_special_tokens=True:\n"
            f"{repr(generated_full_text_skip)}"
        )
        print(
            "generated_text_skip_special_tokens=False:\n"
            f"{repr(generated_full_text_raw)}"
        )
        print(
            f"secret_word_candidate={repr(extract_secret_word_candidate(generated_full_text_skip))}"
        )
        print(
            "is_immediate_im_end_only="
            f"{is_immediate_im_end_only(processor.tokenizer, generated_full_ids)}"
        )


if __name__ == "__main__":
    main()
