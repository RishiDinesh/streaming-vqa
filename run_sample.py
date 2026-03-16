import argparse
from typing import Dict

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.loader import create_video_qa_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more Video QA samples through LLaVA-OneVision-7B and compare predictions with labels."
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="./vnbench_data/VNBench_new/",
        help="Root folder containing video files.",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="./vnbench_data/anno.jsonl",
        help="JSON/JSONL annotation path.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Frames sampled per video.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32000,
        help="Max text length used by dataset preprocessing.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum newly generated tokens.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to run from the dataloader.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def preview_text(text: str, limit: int = 300) -> str:
    compact = text.replace("\n", "\\n")
    if len(compact) <= limit:
        return compact
    head = compact[: limit // 2]
    tail = compact[-(limit // 2) :]
    return f"{head} ... {tail}"


def get_sampling_debug(video_path: str, num_frames: int, needle_time: float | None) -> str:
    try:
        import cv2
    except Exception as exc:
        return f"[Debug] sampling info unavailable (opencv import failed: {exc})"

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if total_frames <= 0 or fps <= 0:
        return (
            "[Debug] sampling info unavailable "
            f"(total_frames={total_frames}, fps={fps})"
        )

    indices = (
        torch.linspace(0, total_frames - 1, steps=num_frames)
        .round()
        .long()
        .tolist()
    )
    times = [idx / fps for idx in indices]
    msg = (
        "[Debug] frame sampling: "
        f"total_frames={total_frames}, fps={fps:.3f}, "
        f"indices={indices}, times_sec={[round(t, 2) for t in times]}"
    )

    if needle_time is not None:
        nearest = min(times, key=lambda t: abs(t - needle_time))
        delta = abs(nearest - needle_time)
        msg += (
            f", needle_time={needle_time:.2f}, "
            f"nearest_sample={nearest:.2f}, delta={delta:.2f}s"
        )

    return msg


def extract_prompt_only_inputs(
    model_inputs: Dict[str, torch.Tensor], labels: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Convert training-formatted batch (prompt+answer) into inference-formatted
    inputs (prompt only), using labels == -100 as the prompt mask.
    """
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError(
            f"Expected labels shape [1, L] for batch_size=1, got {tuple(labels.shape)}"
        )

    answer_positions = torch.nonzero(labels[0] != -100, as_tuple=False)
    if answer_positions.numel() == 0:
        prompt_len = model_inputs["input_ids"].shape[1]
    else:
        prompt_len = int(answer_positions[0].item())

    if prompt_len <= 0:
        raise ValueError("Prompt length resolved to 0; cannot run generation.")

    trimmed_inputs: Dict[str, torch.Tensor] = dict(model_inputs)
    trimmed_inputs["input_ids"] = model_inputs["input_ids"][:, :prompt_len]
    if "attention_mask" in model_inputs:
        trimmed_inputs["attention_mask"] = model_inputs["attention_mask"][:, :prompt_len]
    return trimmed_inputs


def prepare_model_inputs(
    batch: Dict[str, torch.Tensor], device: torch.device, model_dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    model_inputs: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "labels":
            continue
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                model_inputs[key] = value.to(device=device, dtype=model_dtype)
            else:
                model_inputs[key] = value.to(device=device)
        else:
            model_inputs[key] = value
    return model_inputs


def get_dtype_and_device() -> tuple[torch.dtype, torch.device]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"CUDA is available. Running on GPU with {dtype}.")
        return dtype, torch.device("cuda")
    print("CUDA not available. Running on CPU with float32 (this may be very slow).")
    return torch.float32, torch.device("cpu")


def main() -> None:
    args = parse_args()

    dtype, device = get_dtype_and_device()

    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"Loading model: {args.model_id}")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to(device)
    model.eval()

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

    dataloader = create_video_qa_dataloader(
        video_root=args.video_root,
        annotation_path=args.annotation_path,
        processor=processor,
        model_id=args.model_id,
        num_frames=args.num_frames,
        max_length=args.max_length,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    dataset = dataloader.dataset

    print("\nRunning samples...")
    total = 0
    exact_match = 0

    for i, batch in enumerate(dataloader):
        if i >= args.num_samples:
            break

        labels = batch["labels"]
        model_inputs = prepare_model_inputs(batch, device, dtype)
        model_inputs = extract_prompt_only_inputs(model_inputs, labels.to(device))

        input_len = model_inputs["input_ids"].shape[1]
        raw_sample = dataset.samples[i]
        resolved_video_path = dataset._resolve_video_path(raw_sample["video"])
        options = raw_sample.get("options", None)
        needle_time = None
        if isinstance(raw_sample.get("needle_time"), list) and raw_sample["needle_time"]:
            needle_time = float(raw_sample["needle_time"][0])

        print(f"\n[Debug] Sample {i} raw annotation")
        print(f"[Debug] question: {raw_sample['question']}")
        print(f"[Debug] gt      : {raw_sample['gt']}")
        print(f"[Debug] video field in annotation: {raw_sample['video']}")
        print(f"[Debug] resolved video path     : {resolved_video_path}")
        if options is not None:
            print(f"[Debug] options : {options}")
        if needle_time is not None:
            print(f"[Debug] needle_time (sec): {needle_time}")
        print(get_sampling_debug(resolved_video_path, args.num_frames, needle_time))

        print("[Debug] Batch tensor summary")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(
                    f"[Debug] {key}: shape={tuple(value.shape)} dtype={value.dtype} "
                    f"device={value.device}"
                )

        prompt_text = processor.tokenizer.decode(
            model_inputs["input_ids"][0].detach().cpu().tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        print(f"[Debug] prompt token length: {input_len}")
        print(f"[Debug] prompt preview: {preview_text(prompt_text, limit=500)}")
        if options is not None and isinstance(options, list):
            option_presence = [opt in prompt_text for opt in options]
            print(f"[Debug] options present in prompt: {option_presence}")

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        pred_ids = generated_ids[:, input_len:]
        print(
            f"[Debug] generated ids shape={tuple(generated_ids.shape)}, "
            f"new token count={pred_ids.shape[1]}"
        )
        pred_text = processor.tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        label_ids = labels[0][labels[0] != -100]
        label_text = processor.tokenizer.decode(
            label_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        print(f"[Debug] label token count: {label_ids.numel()}")
        print(
            f"[Debug] label token ids (first 32): "
            f"{label_ids[:32].tolist() if label_ids.numel() > 0 else []}"
        )

        pred_norm = normalize_text(pred_text)
        label_norm = normalize_text(label_text)

        is_exact = pred_norm == label_norm
        overlap = bool(pred_norm) and bool(label_norm) and (
            pred_norm in label_norm or label_norm in pred_norm
        )

        total += 1
        exact_match += int(is_exact)

        print(f"\nSample {i}")
        print(f"Prediction: {pred_text}")
        print(f"Label     : {label_text}")
        print(f"Exact match: {is_exact}")
        print(f"Substring overlap: {overlap}")

    if total == 0:
        print("No samples were processed. Check annotation_path/video_root.")
        return

    print("\nSummary")
    print(f"Processed samples: {total}")
    print(f"Exact match count: {exact_match}")
    print(f"Exact match rate : {exact_match / total:.4f}")


if __name__ == "__main__":
    main()
