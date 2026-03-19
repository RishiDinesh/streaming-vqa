import torch.backends.cudnn as cudnn
from typing import Any, Dict, Tuple

import torch
from tqdm import tqdm
from transformers import AutoConfig

from duo_attn.data import create_video_qa_dataloader


def resolve_cuda_device(device_arg) -> torch.device:
    if isinstance(device_arg, list):
        if len(device_arg) == 0:
            raise ValueError("Empty --device list is not supported.")
        device = torch.device(f"cuda:{device_arg[0]}")
    elif device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cpu":
        device = torch.device("cpu")
    elif isinstance(device_arg, str) and device_arg.startswith("cuda"):
        device = torch.device(device_arg)
    else:
        raise ValueError(f"Unsupported device argument: {device_arg}")

    if device.type != "cuda":
        raise ValueError("Efficiency benchmarks require CUDA devices.")
    return device


def is_llava_onevision_model(model_name: str) -> bool:
    name = model_name.lower()
    if "llava_onevision" in name or "llava-onevision" in name:
        return True

    # Fallback for local checkpoints whose directory name may not include llava-onevision.
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
    except Exception:
        return False
    return "llava_onevision" in str(model_type).lower()


def prepare_model_inputs(
    batch: Dict[str, Any],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Dict[str, Any]:
    model_inputs: Dict[str, Any] = {}
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


def extract_prompt_only_inputs(
    model_inputs: Dict[str, Any], labels: torch.Tensor
) -> Tuple[Dict[str, Any], int]:
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError(
            f"Expected labels with shape [1, L] for benchmarking, got {tuple(labels.shape)}."
        )

    answer_positions = torch.nonzero(labels[0] != -100, as_tuple=False)
    if answer_positions.numel() == 0:
        prompt_len = model_inputs["input_ids"].shape[1]
    else:
        prompt_len = int(answer_positions[0].item())
    if prompt_len <= 0:
        raise ValueError("Resolved prompt length is 0, cannot benchmark generation.")

    trimmed_inputs = dict(model_inputs)
    trimmed_inputs["input_ids"] = model_inputs["input_ids"][:, :prompt_len]
    if "attention_mask" in model_inputs:
        trimmed_inputs["attention_mask"] = model_inputs["attention_mask"][:, :prompt_len]
    return trimmed_inputs, prompt_len


def build_llava_first_sample_inputs(
    args,
    processor,
    device: torch.device,
    model_dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataloader = create_video_qa_dataloader(
        video_root=args.video_root,
        annotation_path=args.annotation_path,
        processor=processor,
        model_id=args.model_name,
        num_frames=args.num_frames,
        max_length=args.max_length,
        use_chat_template=not args.disable_video_chat_template,
        answer_prefix=args.video_answer_prefix,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    iterator = iter(dataloader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise ValueError("Video-QA dataloader is empty; cannot benchmark.") from exc

    dataset = dataloader.dataset
    sample_diag: Dict[str, Any] = {
        "sample_index": 0,
        "frames_used": int(args.num_frames),
    }
    if hasattr(dataset, "samples") and len(dataset.samples) > 0:
        raw = dataset.samples[0]
        sample_diag["sample_id"] = str(raw.get("id", raw.get("sample_id", "")))
        sample_diag["video"] = str(raw.get("video", ""))
        sample_diag["question"] = str(raw.get("question", ""))
        if hasattr(dataset, "_resolve_video_path"):
            try:
                sample_diag["resolved_video_path"] = dataset._resolve_video_path(
                    raw.get("video", "")
                )
            except Exception as exc:  # pragma: no cover - best-effort metadata
                sample_diag["resolved_video_path_error"] = str(exc)

    model_inputs = prepare_model_inputs(batch, device, model_dtype)
    labels = batch["labels"].to(device=device)
    prompt_inputs, prompt_len = extract_prompt_only_inputs(model_inputs, labels)
    sample_diag["prompt_token_length"] = int(prompt_len)

    for key, value in prompt_inputs.items():
        if key in {"input_ids", "attention_mask"} or not torch.is_tensor(value):
            continue
        sample_diag[f"{key}_shape"] = tuple(value.shape)

    return prompt_inputs, sample_diag


def bench_func(func, num_steps=100, num_warmup_steps=5):
    cudnn.benchmark = True
    pbar = tqdm(range(num_warmup_steps), desc="Warming up...")
    for _ in pbar:
        func()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(range(num_steps), desc="Benchmarking Latency and Memory...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in pbar:
        func()
    end.record()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)
    avg_time = total_time / num_steps
    print(f"Average latency: {avg_time:.2f} ms")
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    return (
        avg_time,
        peak_memory,
    )
