#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TERM", "xterm-256color")
os.environ.setdefault("FORCE_COLOR", "1")
os.environ.setdefault("CLICOLOR_FORCE", "1")
os.environ.setdefault("PY_COLORS", "1")

import torch
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from tqdm import tqdm
from transformers import logging as transformers_logging
from transformers import AutoConfig, AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.eval.efficiency.prefill_eval_llava import load_video_frames
from duo_attn.eval.validate.inference import build_llava_video_inputs_embeds
from duo_attn.eval.validate.runtime import resolve_device_and_dtype
from duo_attn.patch import enable_duo_attention_eval
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache
from duo_attn.utils import (
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)


DEFAULT_LOCAL_MODEL = "models/llava-hf-llava-onevision-qwen2-7b-ov-hf"
DEFAULT_HF_MODEL = "llava-hf/llava-onevision-qwen2-7b-ov-hf"


def resolve_default_model_name() -> str:
    local_model_path = Path(DEFAULT_LOCAL_MODEL)
    if local_model_path.exists():
        return str(local_model_path)
    return DEFAULT_HF_MODEL


DEFAULT_MODEL = resolve_default_model_name()


class _TransformersNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "passing `past_key_values` as a tuple of tuples" in message:
            return False
        if "Starting from v4.46, the `logits` model output" in message:
            return False
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single Llava-OneVision video prompt with live decode tracing, "
            "prefill/decode timing, and memory diagnostics."
        )
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=22000)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--decode_wrap_lines",
        type=int,
        default=15,
        help="Unused compatibility flag retained for existing commands.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What's in the video? Explain in detail.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="default",
        choices=["default", "eager", "sdpa", "flash_attention_2"],
        help=(
            "Attention backend for baseline mode. 'default' keeps the model's "
            "stock attention selection; DuoAttention mode always forces eager."
        ),
    )
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
        help="Path to an attention pattern directory used for DuoAttention.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used when sparsifying attention heads for DuoAttention.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="Optional target sparsity override for DuoAttention.",
    )
    parser.add_argument(
        "--sink_size",
        type=int,
        default=None,
        help="Optional sink size override for DuoAttention.",
    )
    parser.add_argument(
        "--recent_size",
        type=int,
        default=None,
        help="Optional recent size override for DuoAttention.",
    )
    parser.add_argument(
        "--print_full_text_each_step",
        action="store_true",
        help="Print cumulative decoded text after every generated token.",
    )
    parser.add_argument(
        "--word_stream",
        action="store_true",
        default=True,
        help="Print generated text word by word during decoding.",
    )
    parser.add_argument(
        "--no_word_stream",
        dest="word_stream",
        action="store_false",
        help="Disable word-by-word streaming and show token-level debug lines instead.",
    )
    parser.add_argument(
        "--stream_every",
        type=int,
        default=1,
        help="How often to print latency diagnostics during decoding.",
    )
    parser.add_argument(
        "--report_json",
        type=str,
        default=None,
        help="Optional path to save a full structured report.",
    )
    parser.add_argument(
        "--disable_auto_frame_backoff",
        action="store_true",
        help=(
            "Disable automatic reduction of frame count when the prompt text is "
            "truncated out of the encoded multimodal context."
        ),
    )
    return parser.parse_args()


def print_rule(char: str = "=") -> None:
    print(char * 100)


def print_section(title: str) -> None:
    print()
    print_rule("=")
    print(title)
    print_rule("=")


def format_peak_memory(snapshot: Dict[str, Optional[float]]) -> str:
    peak_bytes = snapshot.get("max_allocated_bytes", None)
    if peak_bytes is None:
        return "n/a"
    return format_mb(peak_bytes)


def print_stage_header(console: Console, title: str, style: str = "bold white") -> None:
    console.print(Rule(Text(title, style=style), style="dim"))


def bytes_to_mb(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def bytes_to_gb(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0 * 1024.0)


def format_mb(num_bytes: float) -> str:
    mb = bytes_to_mb(num_bytes)
    if mb >= 1024.0:
        return f"{mb / 1024.0:.2f} GB"
    return f"{mb:.2f} MB"


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(get_cuda_device_index(device))


def get_cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected CUDA/ROCm device, got {device}.")
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def get_gpu_backend_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "gpu"


def get_gpu_label() -> str:
    backend = get_gpu_backend_name()
    if backend == "rocm":
        return "gpu_memory (rocm/hip via torch.cuda)"
    if backend == "cuda":
        return "gpu_memory (cuda)"
    if backend == "gpu":
        return "gpu_memory"
    return "gpu_memory"


def snapshot_gpu_memory(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "max_allocated_bytes": None,
            "max_reserved_bytes": None,
            "free_bytes": None,
            "total_bytes": None,
        }

    device_index = get_cuda_device_index(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return {
        "allocated_bytes": float(torch.cuda.memory_allocated(device_index)),
        "reserved_bytes": float(torch.cuda.memory_reserved(device_index)),
        "max_allocated_bytes": float(torch.cuda.max_memory_allocated(device_index)),
        "max_reserved_bytes": float(torch.cuda.max_memory_reserved(device_index)),
        "free_bytes": float(free_bytes),
        "total_bytes": float(total_bytes),
    }


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def estimate_past_key_values_bytes(past_key_values: Any) -> int:
    total = 0
    if past_key_values is None:
        return total
    if torch.is_tensor(past_key_values):
        return tensor_bytes(past_key_values)
    if isinstance(past_key_values, (list, tuple)):
        for item in past_key_values:
            total += estimate_past_key_values_bytes(item)
    return total


def summarize_past_key_values(past_key_values: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "layers": 0,
        "entries_per_layer": [],
        "example_shapes": [],
        "estimated_bytes": estimate_past_key_values_bytes(past_key_values),
    }
    if not isinstance(past_key_values, (list, tuple)):
        return summary

    summary["layers"] = len(past_key_values)
    for layer in past_key_values:
        if isinstance(layer, (list, tuple)):
            summary["entries_per_layer"].append(len(layer))
            for item in layer:
                if torch.is_tensor(item) and len(summary["example_shapes"]) < 4:
                    summary["example_shapes"].append(
                        {
                            "shape": list(item.shape),
                            "dtype": str(item.dtype),
                            "device": str(item.device),
                        }
                    )
        elif torch.is_tensor(layer):
            summary["entries_per_layer"].append(1)
            if len(summary["example_shapes"]) < 4:
                summary["example_shapes"].append(
                    {
                        "shape": list(layer.shape),
                        "dtype": str(layer.dtype),
                        "device": str(layer.device),
                    }
                )
    return summary


def token_repr(tokenizer, token_id: int) -> str:
    raw = tokenizer.convert_ids_to_tokens(int(token_id))
    return repr(raw)


def decode_text(tokenizer, token_ids: Sequence[int], skip_special_tokens: bool) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=True,
    )


class WordStreamPrinter:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.generated_ids: List[int] = []
        self.emitted_text = ""
        self.pending_text = ""

    def _decode(self) -> str:
        return self.tokenizer.decode(
            self.generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def _flush_complete_words(self) -> List[str]:
        emitted: List[str] = []
        while True:
            match = re.match(r"^(\S+)(\s+)(.*)$", self.pending_text, flags=re.DOTALL)
            if match is not None:
                word = match.group(1) + match.group(2)
                remainder = match.group(3)
                if word:
                    emitted.append(word)
                self.pending_text = remainder
                continue

            if self.pending_text and self.pending_text[-1] in ".!?;:,":
                emitted.append(self.pending_text)
                self.pending_text = ""
                continue

            break
        return emitted

    def push(self, token_id: int) -> List[str]:
        self.generated_ids.append(int(token_id))
        decoded = self._decode()
        if decoded.startswith(self.emitted_text):
            delta = decoded[len(self.emitted_text) :]
        else:
            delta = decoded
        self.emitted_text = decoded
        if not delta:
            return []
        self.pending_text += delta
        return self._flush_complete_words()

    def flush_partial(self) -> str:
        text = self.pending_text
        self.pending_text = ""
        return text


def visible_delta(previous_text: str, current_text: str) -> str:
    if current_text.startswith(previous_text):
        return current_text[len(previous_text) :]
    return current_text


def build_console() -> Console:
    return Console(
        force_terminal=True,
        force_interactive=True,
        color_system="truecolor",
        soft_wrap=True,
    )


def ensure_generation_pad_token(model, processor) -> None:
    if model.generation_config.pad_token_id is not None:
        return

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        eos_token_id = model.generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            pad_token_id = eos_token_id[0] if eos_token_id else None
        else:
            pad_token_id = eos_token_id
    if pad_token_id is not None:
        model.generation_config.pad_token_id = int(pad_token_id)


def normalize_eos_token_ids(eos_token_id: Any) -> List[int]:
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]
    return [int(token_id) for token_id in eos_token_id]


def processor_outputs_to_device(
    batch: Dict[str, Any],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=model_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def encode_prompt(
    processor,
    frames: Sequence[Any],
    rendered_prompt: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    attempts = [
        {"text": rendered_prompt, "videos": [frames]},
        {"text": [rendered_prompt], "videos": [frames]},
        {"text": rendered_prompt, "videos": frames},
        {"text": [rendered_prompt], "videos": frames},
    ]
    last_error = None
    for kwargs in attempts:
        for with_length_controls in (True, False):
            try:
                processor_kwargs: Dict[str, Any] = {
                    "return_tensors": "pt",
                    **kwargs,
                }
                if with_length_controls:
                    processor_kwargs["truncation"] = True
                    processor_kwargs["max_length"] = int(max_length)
                outputs = processor(**processor_kwargs)
                if outputs.get("input_ids", None) is not None:
                    return dict(outputs)
            except Exception as exc:
                last_error = exc
    raise RuntimeError("Unable to encode video/text prompt.") from last_error


def print_memory_snapshot(label: str, snapshot: Dict[str, Optional[float]]) -> None:
    print(f"{label}:")
    if snapshot["allocated_bytes"] is None:
        print("  gpu_memory=unavailable (non-GPU run)")
        return
    print(f"  backend={get_gpu_backend_name()}")
    print(f"  allocated={format_mb(snapshot['allocated_bytes'])}")
    print(f"  reserved={format_mb(snapshot['reserved_bytes'])}")
    print(f"  max_allocated={format_mb(snapshot['max_allocated_bytes'])}")
    print(f"  max_reserved={format_mb(snapshot['max_reserved_bytes'])}")
    print(f"  free={format_mb(snapshot['free_bytes'])}")
    print(f"  total={format_mb(snapshot['total_bytes'])}")


def memory_summary_text(snapshot: Dict[str, Optional[float]]) -> str:
    if snapshot["allocated_bytes"] is None:
        return "n/a"
    return format_mb(snapshot["allocated_bytes"])


def silence_runtime_noise() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*passing `past_key_values` as a tuple of tuples.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*passing `past_key_values` as a tuple of tuples.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Starting from v4\.46, the `logits` model output.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*Starting from v4\.46, the `logits` model output.*",
    )
    logging.getLogger("transformers").addFilter(_TransformersNoiseFilter())
    logging.getLogger("transformers").setLevel(logging.ERROR)
    transformers_logging.set_verbosity_error()
    transformers_logging.get_logger().addFilter(_TransformersNoiseFilter())
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def write_report(path: str, report: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def normalize_text_for_match(text: str) -> str:
    return " ".join(str(text).lower().split())


def prompt_text_survived(question_text: str, decoded_raw: str, decoded_skip: str) -> bool:
    target = normalize_text_for_match(question_text)
    if not target:
        return True
    return target in normalize_text_for_match(decoded_raw) or target in normalize_text_for_match(decoded_skip)


def render_generation_prompt(processor, prompt_text: str) -> str:
    multimodal_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    errors: List[str] = []

    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                multimodal_conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            errors.append(f"processor.apply_chat_template: {type(exc).__name__}: {exc}")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            video_token = getattr(processor, "video_token", None) or "<video>"
            text_conversation = [
                {
                    "role": "user",
                    "content": f"{video_token}\n{prompt_text}",
                }
            ]
            return tokenizer.apply_chat_template(
                text_conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            errors.append(f"tokenizer.apply_chat_template: {type(exc).__name__}: {exc}")

    try:
        video_token = getattr(processor, "video_token", None) or "<video>"
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{video_token}\n{prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    except Exception as exc:
        errors.append(f"manual_prompt: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Unable to render a chat-template prompt for this model. "
        + " | ".join(errors if errors else ["No chat template API available"])
    )


def build_decode_renderable(
    console: Console,
    visible_text: str,
    mem_text: str,
    avg_latency_ms: Optional[float],
    tok_per_s: float,
) -> Group:
    latency_text = f"{avg_latency_ms:.2f}" if avg_latency_ms is not None else "n/a"
    header = Text()
    header.append("Decoding", style="bold blue")
    header.append(" (", style="bold white")
    header.append("Mem", style="bold magenta")
    header.append(f": {mem_text}", style="white")
    header.append(" | ", style="dim white")
    header.append("Latency", style="bold yellow")
    header.append(f": {latency_text} ms/tok", style="white")
    header.append(" | ", style="dim white")
    header.append("Tok/s", style="bold green")
    header.append(f": {tok_per_s:.2f}", style="white")
    header.append("):", style="bold white")
    width = max(40, console.size.width)
    wrapped_lines: List[str] = []
    available_width = max(20, width)
    text = visible_text.rstrip()
    if text:
        paragraphs = text.splitlines() or [text]
        for paragraph in paragraphs:
            if not paragraph.strip():
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    paragraph,
                    width=available_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [paragraph]
            )
    else:
        wrapped_lines.append("")
    body = "\n".join(wrapped_lines)
    return Group(header, Text(body, style="white"))


def prepare_prompt_with_backoff(
    processor,
    video_path: str,
    requested_num_frames: int,
    prompt_text: str,
    rendered_prompt: str,
    max_length: int,
    auto_backoff: bool,
) -> Tuple[List[Any], float, int, Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    attempt_log: List[Dict[str, Any]] = []
    num_frames = max(1, int(requested_num_frames))

    while True:
        frames, clip_seconds = load_video_frames(
            video_path=video_path,
            num_frames=num_frames,
            prefix_ratio=1.0,
        )
        raw_inputs = encode_prompt(
            processor=processor,
            frames=frames,
            rendered_prompt=rendered_prompt,
            max_length=int(max_length),
        )
        input_ids = raw_inputs["input_ids"][0].detach().cpu().tolist()
        decoded_raw = decode_text(processor.tokenizer, input_ids, False)
        decoded_skip = decode_text(processor.tokenizer, input_ids, True)
        # Compare against the user-authored text rather than the fully rendered
        # chat template. Special tokens and expanded video placeholders mean the
        # decoded prompt will not round-trip to the exact rendered template.
        survived = prompt_text_survived(prompt_text, decoded_raw, decoded_skip)
        reached_cap = len(input_ids) >= int(max_length)

        attempt = {
            "num_frames": num_frames,
            "prompt_token_count": len(input_ids),
            "reached_max_length_cap": reached_cap,
            "prompt_text_survived": survived,
            "decoded_skip_preview": decoded_skip[:200],
            "decoded_raw_preview": decoded_raw[:200],
        }
        attempt_log.append(attempt)

        if survived or not auto_backoff or num_frames <= 1:
            return frames, clip_seconds, num_frames, dict(raw_inputs), attempt_log

        next_num_frames = max(1, num_frames // 2)
        if next_num_frames == num_frames:
            return frames, clip_seconds, num_frames, dict(raw_inputs), attempt_log
        num_frames = next_num_frames


def resolve_attention_mode(args: argparse.Namespace) -> str:
    attention_mode = str(args.attention_mode)
    if attention_mode == "auto":
        attention_mode = "duo" if args.attn_load_dir is not None else "baseline"
    if attention_mode == "duo" and args.attn_load_dir is None:
        raise ValueError("--attn_load_dir is required when --attention_mode=duo.")
    return attention_mode


def resolve_effective_attn_implementation(
    attention_mode: str,
    requested_attn_implementation: str,
) -> Optional[str]:
    if attention_mode == "duo":
        return "eager"
    if requested_attn_implementation == "default":
        return None
    return requested_attn_implementation


def configure_model_for_attention_mode(
    model,
    attention_mode: str,
    attn_load_dir: Optional[str],
    threshold: float,
    sparsity: Optional[float],
    sink_size_override: Optional[int],
    recent_size_override: Optional[int],
) -> Dict[str, Any]:
    resolved_sparsity = 0.0
    resolved_sink_size: Optional[int] = None
    resolved_recent_size: Optional[int] = None

    if attention_mode == "duo":
        if attn_load_dir is None:
            raise ValueError("--attn_load_dir is required when --attention_mode=duo.")
        full_attention_heads, sink_size, recent_size = load_attn_pattern(attn_load_dir)
        if sink_size_override is not None:
            sink_size = sink_size_override
        if recent_size_override is not None:
            recent_size = recent_size_override
        full_attention_heads, resolved_sparsity = sparsify_attention_heads(
            full_attention_heads,
            threshold,
            sparsity,
        )
        enable_duo_attention_eval(model, full_attention_heads, sink_size, recent_size)
        resolved_sink_size = int(sink_size)
        resolved_recent_size = int(recent_size)
    else:
        enable_tuple_kv_cache(model)

    return {
        "attention_mode": attention_mode,
        "attn_load_dir": os.path.abspath(attn_load_dir) if attn_load_dir else None,
        "resolved_sparsity": float(resolved_sparsity),
        "sink_size": resolved_sink_size,
        "recent_size": resolved_recent_size,
    }
def main() -> None:
    args = parse_args()
    silence_runtime_noise()
    seed_everything(args.seed)
    device, dtype = resolve_device_and_dtype(args)
    attention_mode = resolve_attention_mode(args)
    requested_attn_implementation = str(args.attn_implementation)
    effective_attn_implementation = resolve_effective_attn_implementation(
        attention_mode,
        requested_attn_implementation
    )

    video_path = os.path.abspath(args.video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(get_cuda_device_index(device))

    startup_memory = snapshot_gpu_memory(device)

    model_load_t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model_load_kwargs: Dict[str, Any] = {
        "config": config,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if effective_attn_implementation is not None:
        model_load_kwargs["attn_implementation"] = effective_attn_implementation
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_name,
        **model_load_kwargs,
    )
    model.eval()
    loaded_attn_implementation = getattr(model.config, "_attn_implementation", "unknown")
    attention_summary = configure_model_for_attention_mode(
        model=model,
        attention_mode=attention_mode,
        attn_load_dir=args.attn_load_dir,
        threshold=args.threshold,
        sparsity=args.sparsity,
        sink_size_override=args.sink_size,
        recent_size_override=args.recent_size,
    )
    model.to(device)
    ensure_generation_pad_token(model, processor)
    synchronize_if_needed(device)
    model_load_ms = (time.perf_counter() - model_load_t0) * 1000.0
    post_model_load_memory = snapshot_gpu_memory(device)

    render_prompt_t0 = time.perf_counter()
    rendered_prompt = render_generation_prompt(processor, args.prompt)
    render_prompt_ms = (time.perf_counter() - render_prompt_t0) * 1000.0

    frame_load_t0 = time.perf_counter()
    frames, clip_seconds, used_num_frames, raw_inputs, prompt_attempts = prepare_prompt_with_backoff(
        processor=processor,
        video_path=video_path,
        requested_num_frames=int(args.num_frames),
        prompt_text=args.prompt,
        rendered_prompt=rendered_prompt,
        max_length=int(args.max_length),
        auto_backoff=not args.disable_auto_frame_backoff,
    )
    frame_load_ms = (time.perf_counter() - frame_load_t0) * 1000.0

    move_inputs_t0 = time.perf_counter()
    model_inputs = processor_outputs_to_device(raw_inputs, device, dtype)
    synchronize_if_needed(device)
    move_inputs_ms = (time.perf_counter() - move_inputs_t0) * 1000.0
    post_input_prep_memory = snapshot_gpu_memory(device)

    prompt_input_ids = model_inputs["input_ids"][0].detach().cpu().tolist()
    prompt_text_raw = decode_text(processor.tokenizer, prompt_input_ids, False)
    prompt_text_skip = decode_text(processor.tokenizer, prompt_input_ids, True)
    embed_t0 = time.perf_counter()
    with torch.inference_mode():
        inputs_embeds = build_llava_video_inputs_embeds(model, model_inputs)
    synchronize_if_needed(device)
    embed_ms = (time.perf_counter() - embed_t0) * 1000.0
    inputs_embeds_bytes = tensor_bytes(inputs_embeds)
    post_embed_memory = snapshot_gpu_memory(device)
    demo_console = build_console()
    banner_text = (
        "MULTIMODAL DUO ATTENTION"
        if attention_summary["attention_mode"] == "duo"
        else "FULL ATTENTION"
    )
    banner_style = "bold green" if attention_summary["attention_mode"] == "duo" else "bold red"
    demo_console.print(Rule(Text(banner_text, style=banner_style), style=banner_style))
    demo_console.print(
        Text.assemble(
            ("Attention mode/backend", "bold cyan"),
            (
                f": {attention_summary['attention_mode']} / "
                f"{loaded_attn_implementation}",
                "white",
            ),
        )
    )
    print_stage_header(demo_console, "INPUT", "bold cyan")
    demo_console.print(
        Text.assemble(
            ("Frames requested/used", "bold cyan"),
            (f": {int(args.num_frames)} / {int(used_num_frames)}", "white"),
        )
    )
    demo_console.print(
        Text.assemble(
            ("Input sequence length", "bold cyan"),
            (f": {inputs_embeds.shape[1]}", "white"),
        )
    )
    demo_console.print()
    print_stage_header(demo_console, "PREFILLING", "bold yellow")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(get_cuda_device_index(device))
    prefill_mem_before = snapshot_gpu_memory(device)
    prefill_bar = tqdm(
        total=1,
        desc=f"Pre-filling ({inputs_embeds.shape[1]}/{inputs_embeds.shape[1]}, Mem: {memory_summary_text(prefill_mem_before)})",
        unit="step",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.25,
        smoothing=0.0,
    )
    prefill_t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
        )
    synchronize_if_needed(device)
    prefill_ms = (time.perf_counter() - prefill_t0) * 1000.0
    prefill_mem_after = snapshot_gpu_memory(device)
    past_key_values = outputs.past_key_values
    first_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    kv_summary = summarize_past_key_values(past_key_values)
    prefill_bar.set_description(
        f"Pre-filling ({inputs_embeds.shape[1]}/{inputs_embeds.shape[1]}, Mem: {memory_summary_text(prefill_mem_after)})"
    )
    prefill_bar.set_postfix(
        {
            "time_s": f"{prefill_ms / 1000.0:.2f}",
            "tok/s": f"{len(prompt_input_ids) / max(prefill_ms / 1000.0, 1e-9):.2f}",
        },
        refresh=False,
    )
    prefill_bar.update(1)
    prefill_bar.close()
    demo_console.print(
        Text.assemble(
            ("Pre-filling time", "bold yellow"),
            (f": {prefill_ms / 1000.0:.2f}s", "white"),
        )
    )
    demo_console.print(
        Text.assemble(
            ("Pre-filling memory", "bold magenta"),
            (f": {format_peak_memory(prefill_mem_after)}", "white"),
        )
    )
    demo_console.print()
    eos_token_ids = set(normalize_eos_token_ids(model.generation_config.eos_token_id))
    generated_ids: List[int] = []
    step_records: List[Dict[str, Any]] = []
    decode_started_at = time.perf_counter()
    previous_visible_text = ""
    avg_recent_latency: Optional[float] = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(get_cuda_device_index(device))

    console = build_console()
    latest_mem_text = memory_summary_text(prefill_mem_after)
    latest_toks_per_s = 0.0
    final_peak_memory_text = latest_mem_text
    force_plain_word_stream = os.environ.get("LIVE_LLAVA_FORCE_PLAIN_STREAM", "0") == "1"
    use_live_word_stream = bool(args.word_stream and not force_plain_word_stream)
    plain_word_stream = (
        WordStreamPrinter(processor.tokenizer)
        if args.word_stream and force_plain_word_stream
        else None
    )

    print_stage_header(console, "DECODING", "bold blue")

    if use_live_word_stream:
        console.print()
        live = Live(
            build_decode_renderable(
                console,
                "",
                latest_mem_text,
                avg_recent_latency,
                latest_toks_per_s,
            ),
            console=console,
            refresh_per_second=12,
            transient=False,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        live.__enter__()
    else:
        live = None
        if plain_word_stream is not None:
            console.print()

    try:
        for step_idx in range(int(args.max_new_tokens)):
            current_token_id = int(first_token[0, 0].item())
            generated_ids.append(current_token_id)
            visible_text = decode_text(processor.tokenizer, generated_ids, True)
            visible_text_raw = decode_text(processor.tokenizer, generated_ids, False)
            delta_text = visible_delta(previous_visible_text, visible_text)
            previous_visible_text = visible_text

            step_mem_before = snapshot_gpu_memory(device)
            step_t0 = time.perf_counter()
            reached_eos = current_token_id in eos_token_ids

            if not reached_eos and step_idx + 1 < int(args.max_new_tokens):
                with torch.inference_mode():
                    outputs = model.language_model(
                        input_ids=first_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                synchronize_if_needed(device)
                step_ms = (time.perf_counter() - step_t0) * 1000.0
                past_key_values = outputs.past_key_values
                first_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                next_token_id = int(first_token[0, 0].item())
            else:
                synchronize_if_needed(device)
                step_ms = (time.perf_counter() - step_t0) * 1000.0
                next_token_id = None

            step_mem_after = snapshot_gpu_memory(device)
            kv_bytes = estimate_past_key_values_bytes(past_key_values)
            elapsed_decode_ms = (time.perf_counter() - decode_started_at) * 1000.0
            generated_token_count = len(generated_ids)
            tokens_per_second = generated_token_count / max(elapsed_decode_ms / 1000.0, 1e-9)
            latest_mem_text = memory_summary_text(step_mem_after)
            latest_toks_per_s = tokens_per_second
            if step_mem_after["max_allocated_bytes"] is not None:
                final_peak_memory_text = format_mb(step_mem_after["max_allocated_bytes"])

            step_record = {
                "step": step_idx + 1,
                "token_id": current_token_id,
                "token_repr": token_repr(processor.tokenizer, current_token_id),
                "delta_text": delta_text,
                "visible_text": visible_text,
                "visible_text_raw": visible_text_raw,
                "latency_ms": step_ms,
                "elapsed_decode_ms": elapsed_decode_ms,
                "tokens_per_second_running": tokens_per_second,
                "kv_cache_estimated_bytes": kv_bytes,
                "memory_before": step_mem_before,
                "memory_after": step_mem_after,
                "reached_eos": reached_eos,
                "next_token_id": next_token_id,
            }
            step_records.append(step_record)

            measured_latencies = [
                record["latency_ms"] for record in step_records if record["next_token_id"] is not None
            ]
            avg_recent_latency = (
                sum(measured_latencies) / len(measured_latencies)
                if measured_latencies
                else None
            )

            if args.word_stream and live is not None:
                live.update(
                    build_decode_renderable(
                        console,
                        visible_text,
                        latest_mem_text,
                        avg_recent_latency,
                        tokens_per_second,
                    ),
                    refresh=True,
                )
                live.refresh()
            elif plain_word_stream is not None:
                streamed_chunks = plain_word_stream.push(current_token_id)
                if streamed_chunks:
                    sys.stdout.write("".join(streamed_chunks))
                    sys.stdout.flush()
            elif not args.word_stream and step_idx % max(1, int(args.stream_every)) == 0:
                print_rule("-")
                print(
                    f"step={step_record['step']} "
                    f"token_id={current_token_id} "
                    f"token={step_record['token_repr']} "
                    f"latency_ms={step_ms:.2f} "
                    f"running_toks_per_s={tokens_per_second:.2f}"
                )
                print(f"delta_text={repr(delta_text)}")
                if args.print_full_text_each_step:
                    print(f"visible_text={repr(visible_text)}")
                if step_mem_after["allocated_bytes"] is not None:
                    print(
                        "memory_after="
                        f"allocated:{format_mb(step_mem_after['allocated_bytes'])} "
                        f"reserved:{format_mb(step_mem_after['reserved_bytes'])} "
                        f"max_allocated:{format_mb(step_mem_after['max_allocated_bytes'])}"
                    )
                    print(f"estimated_kv_cache={format_mb(kv_bytes)}")

            if reached_eos:
                break
    finally:
        if live is not None:
            live.__exit__(None, None, None)
        if plain_word_stream is not None:
            trailing_text = plain_word_stream.flush_partial()
            if trailing_text:
                sys.stdout.write(trailing_text)
                sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

    decode_total_ms = (time.perf_counter() - decode_started_at) * 1000.0
    decode_mem_after = snapshot_gpu_memory(device)
    if decode_mem_after["max_allocated_bytes"] is not None:
        final_peak_memory_text = format_peak_memory(decode_mem_after)
    final_visible_text = decode_text(processor.tokenizer, generated_ids, True).strip()
    final_visible_text_raw = decode_text(processor.tokenizer, generated_ids, False).strip()
    non_terminal_step_latencies = [
        record["latency_ms"]
        for record in step_records
        if record["next_token_id"] is not None
    ]
    avg_decode_ms = (
        sum(non_terminal_step_latencies) / len(non_terminal_step_latencies)
        if non_terminal_step_latencies
        else None
    )

    if args.word_stream:
        console.print()
        console.print(Rule(style="dim"))
        latency_summary = (
            f"{avg_decode_ms:.2f} ms"
            if avg_decode_ms is not None
            else "n/a"
        )
        console.print(
            Text.assemble(
                ("Per-token decoding latency", "bold yellow"),
                (f": {latency_summary}", "white"),
            )
        )
        console.print(
            Text.assemble(
                ("Peak memory", "bold magenta"),
                (f": {final_peak_memory_text}", "white"),
            )
        )

    report: Dict[str, Any] = {
        "args": vars(args),
        "video_path": video_path,
        "device": str(device),
        "gpu_backend": get_gpu_backend_name(),
        "gpu_label": get_gpu_label(),
        "dtype": str(dtype),
        "attn_implementation": loaded_attn_implementation,
        "requested_attention_mode": str(args.attention_mode),
        "attention_mode": attention_summary["attention_mode"],
        "attn_load_dir": attention_summary["attn_load_dir"],
        "resolved_sparsity": attention_summary["resolved_sparsity"],
        "sink_size": attention_summary["sink_size"],
        "recent_size": attention_summary["recent_size"],
        "requested_attn_implementation": requested_attn_implementation,
        "effective_attn_implementation": loaded_attn_implementation,
        "model_type": config.model_type,
        "startup_memory": startup_memory,
        "post_model_load_memory": post_model_load_memory,
        "post_input_prep_memory": post_input_prep_memory,
        "post_embed_memory": post_embed_memory,
        "prefill_memory_before": prefill_mem_before,
        "prefill_memory_after": prefill_mem_after,
        "decode_memory_after": decode_mem_after,
        "model_load_ms": model_load_ms,
        "frame_load_ms": frame_load_ms,
        "render_prompt_ms": render_prompt_ms,
        "move_inputs_ms": move_inputs_ms,
        "decoded_clip_seconds": clip_seconds,
        "frames_loaded": len(frames),
        "requested_num_frames": int(args.num_frames),
        "used_num_frames": int(used_num_frames),
        "prompt_attempts": prompt_attempts,
        "rendered_prompt": rendered_prompt,
        "prompt_token_count": len(prompt_input_ids),
        "prompt_text_skip_special": prompt_text_skip,
        "prompt_text_raw": prompt_text_raw,
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "inputs_embeds_dtype": str(inputs_embeds.dtype),
        "inputs_embeds_estimated_bytes": inputs_embeds_bytes,
        "embed_build_ms": embed_ms,
        "prefill_ms": prefill_ms,
        "prefill_tokens_per_second": len(prompt_input_ids) / max(prefill_ms / 1000.0, 1e-9),
        "prefill_kv_summary": kv_summary,
        "decode_total_ms": decode_total_ms,
        "avg_decode_step_ms": avg_decode_ms,
        "decode_tokens_per_second": len(generated_ids) / max(decode_total_ms / 1000.0, 1e-9),
        "generated_token_ids": generated_ids,
        "generated_text": final_visible_text,
        "generated_text_raw": final_visible_text_raw,
        "steps": step_records,
    }

    if args.report_json:
        write_report(args.report_json, report)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
