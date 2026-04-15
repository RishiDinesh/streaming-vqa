#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from duo_attn.utils import sparsify_attention_heads
from duo_attn.patch.attn_compat import FLASH_ATTN_AVAILABLE
from duo_attn.patch.flashinfer_utils import flashinfer
from duo_attn.patch.streaming_attn import is_blocksparse_available

from streaming.ReKV.datasets import RVS_DATASET_CONFIGS, build_dataset_from_args
from streaming.ReKV.methods import (
    MethodAnswer,
    build_method_backend_report,
    resolve_device,
    resolve_dtype,
)
from streaming.ReKV.run_eval import (
    build_evaluation_manifest,
    evaluate_samples,
    normalize_result_payload_schema,
    validate_resume_payload,
    write_json_atomic,
)

DEFAULT_MODEL = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
DEFAULT_DUO_ATTN_DIR = (
    "outputs/train/"
    "0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632"
)
DEFAULT_DUO_HEADS_FILE = (
    "outputs/train/"
    "0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632/"
    "full_attention_heads_latest.tsv"
)
DEFAULT_STREAMINGTOM_ROOT = "streaming/StreamingTom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "StreamingTOM full-eval runner (chunkable) with run_eval-compatible JSON output."
        )
    )
    parser.add_argument("--dataset", default="rvs_ego", choices=sorted(RVS_DATASET_CONFIGS))
    parser.add_argument("--annotation-path", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--hf-repo-id", default="Becomebright/RVS")
    parser.add_argument("--allow-hf-video-download", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--method",
        default="streamingtom",
        choices=["streamingtom", "duo_plus_streamingtom"],
    )
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video-offset", type=int, default=0)
    parser.add_argument("--video-index", type=int, default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--subsample-name", default=None)
    parser.add_argument("--max-conversations-per-video", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video-decode-threads", type=int, default=1)
    parser.add_argument("--clear-cuda-cache-on-reset", action="store_true")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--flush-every-videos", type=int, default=1)
    parser.add_argument("--flush-every-conversations", type=int, default=1)
    parser.add_argument("--disable-progress-bar", action="store_true")

    parser.add_argument("--streamingtom-root", default=DEFAULT_STREAMINGTOM_ROOT)

    parser.add_argument("--duo-attn-dir", default=DEFAULT_DUO_ATTN_DIR)
    parser.add_argument("--duo-heads-file", default=DEFAULT_DUO_HEADS_FILE)
    parser.add_argument("--duo-threshold", type=float, default=0.5)
    parser.add_argument("--duo-sparsity", type=float, default=0.75)
    parser.add_argument("--duo-sink-size", type=int, default=256)
    parser.add_argument("--duo-recent-size", type=int, default=512)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slugify(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")


def default_output_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(args.model)
    base_dir = Path("outputs") / "evaluations_streaming" / args.dataset.replace("_", "-")
    if args.subsample_name:
        base_dir = base_dir / slugify(args.subsample_name)
    elif args.max_videos is not None:
        base_dir = base_dir / f"subsample{args.max_videos}"
    if int(args.num_chunks) > 1:
        base_dir = base_dir / "shards" / f"chunk{int(args.chunk_index):03d}-of-{int(args.num_chunks):03d}"
    return base_dir / args.method / model_slug / f"{timestamp}_results.json"


def set_streamingtom_env_defaults() -> None:
    defaults = {
        "WRAPPER": "streamingtom",
        "CTR_K": "7",
        "CTR_BETA": "0.6",
        "CTR_SIMILARITY_THRESHOLD": "0.9",
        "CTR_RETAIN_TOKENS": "50",
        "OQM_GROUP_SIZE": "50",
        "OQM_SLIDING_WINDOW_SIZE": "4800",
        "OQM_RETRIEVAL_MAX_TOKENS": "12544",
        "OQM_ENABLE_QUANTIZATION": "1",
        "OQM_QUANTIZATION_BITS": "4",
        "OQM_INIT_TOKEN_COUNT": "14",
        "STREAMING_ENCODER_BATCH_SIZE": "32",
        "STREAMINGTOM_USE_FULL_PROMPT": "0",
        # Enable first-token timing inside StreamingTOM query path.
        "STREAMINGTOM_MEASURE_TTFT": "1",
        # Bound per-question visual context to avoid OOM on long videos.
        "STREAMINGTOM_MAX_FRAMES_CONTEXT": "64",
        # Batch frame preprocessing to limit host/GPU memory spikes.
        "STREAMINGTOM_FRAME_PREPROCESS_BATCH": "16",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def ensure_streamingtom_import(streamingtom_root: Path) -> None:
    llava_next_path = streamingtom_root / "LLaVA-NeXT"
    legacy_core_path = streamingtom_root / "streamingtom-core"
    package_root = streamingtom_root

    if not llava_next_path.is_dir():
        raise FileNotFoundError(f"Missing LLaVA-NeXT path: {llava_next_path}")

    if legacy_core_path.is_dir():
        core_import_path = legacy_core_path
    elif (package_root / "streamingtom").is_dir():
        core_import_path = package_root
    else:
        raise FileNotFoundError(
            "Missing StreamingTOM python package. Expected either "
            f"{legacy_core_path} or {package_root / 'streamingtom'}"
        )

    sys.path.insert(0, str(llava_next_path))
    sys.path.insert(0, str(core_import_path))


def _resolve_cache_root(args: argparse.Namespace) -> Path:
    root = Path(args.streamingtom_root).expanduser().resolve(strict=False)
    if root.is_dir():
        return root
    return (Path.cwd() / args.streamingtom_root).resolve(strict=False)


def _apply_duo_routing(
    model: torch.nn.Module,
    *,
    attn_dir: str,
    heads_file: str,
    threshold: float,
    sparsity: float,
    deploy_sink_size: int,
    deploy_recent_size: int,
) -> dict[str, Any]:
    from duo_attn.patch.llava_onevision import (
        _enable_qwen2_layers_duo_attention_eval,
        _get_qwen2_layers,
    )

    attn_dir_path = Path(attn_dir).expanduser().resolve(strict=False)
    heads_path = Path(heads_file).expanduser().resolve(strict=False)
    if not heads_path.is_file():
        latest = attn_dir_path / "full_attention_heads_latest.tsv"
        default = attn_dir_path / "full_attention_heads.tsv"
        heads_path = latest if latest.is_file() else default
    if not heads_path.is_file():
        raise FileNotFoundError(f"Duo heads file not found: {heads_path}")

    full_attention_heads = np.loadtxt(heads_path, delimiter="	")
    if full_attention_heads.ndim == 1:
        full_attention_heads = np.expand_dims(full_attention_heads, axis=0)

    full_attention_heads, actual_sparsity = sparsify_attention_heads(
        full_attention_heads,
        threshold=threshold,
        sparsity=sparsity,
    )
    full_attention_heads_t = torch.as_tensor(full_attention_heads, dtype=torch.float32)

    layers = _get_qwen2_layers(model)
    if not layers:
        raise ValueError(f"Could not find Qwen2 layers on model type {type(model)}")

    if len(layers) != int(full_attention_heads_t.shape[0]):
        raise ValueError(
            f"Duo heads layer count mismatch: model={len(layers)} vs pattern={int(full_attention_heads_t.shape[0])}"
        )

    # ReKV-style Duo integration:
    # 1) apply official Duo layer reordering + full_attention_heads registration
    # 2) restore StreamingTom's custom attention forward to preserve retrieval path
    # This avoids tuple-KV cache patching semantics while still aligning Duo head routing prep.
    original_forwards = [layer.self_attn.forward for layer in layers]
    _enable_qwen2_layers_duo_attention_eval(
        layers,
        full_attention_heads_t.detach().cpu().numpy(),
        int(deploy_sink_size),
        int(deploy_recent_size),
    )
    for layer, original_forward in zip(layers, original_forwards):
        attn = layer.self_attn
        attn.forward = original_forward
        attn.duo_enable = True
        attn.duo_full_attention_heads = attn.full_attention_heads
        attn.duo_sink_size = int(deploy_sink_size)
        attn.duo_recent_size = int(deploy_recent_size)

    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        if not hasattr(attn, "k_proj"):
            raise ValueError(f"Layer {layer_idx} self_attn is not compatible with duo routing")

        num_kv = attn.k_proj.weight.shape[0] // attn.head_dim
        layer_heads = full_attention_heads_t[layer_idx]
        if int(layer_heads.numel()) != int(num_kv):
            raise ValueError(
                f"Layer {layer_idx} kv-head mismatch: pattern={int(layer_heads.numel())} model={int(num_kv)}"
            )

        # Keep explicit attrs used by StreamingTom's custom Qwen2 attention path.
        attn.duo_full_attention_heads = layer_heads.to(
            device=attn.k_proj.weight.device,
            dtype=attn.k_proj.weight.dtype,
        )

    return {
        "attn_dir": str(attn_dir_path),
        "heads_file": str(heads_path),
        "actual_sparsity": float(actual_sparsity),
        "sink_size": int(deploy_sink_size),
        "recent_size": int(deploy_recent_size),
    }


@dataclass
class StreamingTomMethod:
    pretrained: str
    method_name: str
    device: torch.device
    dtype: torch.dtype
    max_new_tokens: int
    streamingtom_root: str
    duo_enabled: bool = False
    duo_attn_dir: str | None = None
    duo_heads_file: str | None = None
    duo_threshold: float = 0.5
    duo_sparsity: float = 0.75
    duo_sink_size: int = 256
    duo_recent_size: int = 512
    clear_cuda_cache_on_reset: bool = False

    def __post_init__(self) -> None:
        self.frames_ingested = 0
        self.ingested_timestamps_sec: list[float] = []
        self.ingest_latencies_sec: list[float] = []
        self._last_h = 224
        self._last_w = 224
        self._duo_stats: dict[str, Any] | None = None
        self.attn_impl_requested = "flash_attention_2" if FLASH_ATTN_AVAILABLE else "sdpa"
        self.attn_impl_used = self.attn_impl_requested
        self.runtime_backend_info = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device_type": self.device.type,
            "device": str(self.device),
            "accelerator_backend": ("cuda" if self.device.type == "cuda" and torch.cuda.is_available() else "cpu"),
            "device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            "torch_dtype": str(self.dtype).replace("torch.", ""),
            "cuda_version": getattr(torch.version, "cuda", None),
            "flash_attn_available": bool(FLASH_ATTN_AVAILABLE),
            "block_sparse_attn_available": bool(is_blocksparse_available()),
            "flashinfer_available": bool(flashinfer is not None),
            "attention_module_load_path": (
                "flash_attn" if self.attn_impl_requested == "flash_attention_2" else "sdpa_fallback"
            ),
        }

        set_streamingtom_env_defaults()
        max_frames_raw = os.environ.get("STREAMINGTOM_MAX_FRAMES_CONTEXT", "64").strip()
        self.max_frames_context = int(max_frames_raw) if max_frames_raw and int(max_frames_raw) > 0 else None
        batch_raw = os.environ.get("STREAMINGTOM_FRAME_PREPROCESS_BATCH", "16").strip()
        self.frame_preprocess_batch = max(1, int(batch_raw)) if batch_raw else 16
        # Keep only bounded recent frames in RAM to avoid host-memory OOM on long videos.
        self._frames = deque(maxlen=self.max_frames_context)
        ensure_streamingtom_import(_resolve_cache_root(argparse.Namespace(streamingtom_root=self.streamingtom_root)))

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from streamingtom.main import streamingtom as apply_streamingtom

        model_name = get_model_name_from_path(self.pretrained)
        dtype_name = "float16" if self.dtype == torch.float16 else "bfloat16"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.pretrained,
            None,
            model_name,
            device_map=("auto" if self.device.type == "cuda" else "cpu"),
            multimodal=True,
            attn_implementation=self.attn_impl_requested,
            torch_dtype=dtype_name,
        )

        model.tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.model = apply_streamingtom(model, "llava")
        self.image_processor = image_processor

        if self.duo_enabled:
            self._duo_stats = _apply_duo_routing(
                self.model,
                attn_dir=str(self.duo_attn_dir),
                heads_file=str(self.duo_heads_file),
                threshold=float(self.duo_threshold),
                sparsity=float(self.duo_sparsity),
                deploy_sink_size=int(self.duo_sink_size),
                deploy_recent_size=int(self.duo_recent_size),
            )

        self._streaming_video_id: str | None = None
        self._streaming_is_start = True

    def get_evaluation_manifest(self) -> dict[str, Any]:
        backend_resolution = build_method_backend_report(
            runtime_backend_info=self.runtime_backend_info,
        )
        resolved_attn_backend = (
            "flash_attn"
            if self.attn_impl_used == "flash_attention_2"
            else ("sdpa" if self.attn_impl_used == "sdpa" else str(self.attn_impl_used))
        )
        backend_resolution["full_attn_backend_actual"] = (
            "flash_attn" if self.attn_impl_used == "flash_attention_2" else "sdpa_fallback"
        )
        backend_resolution["streaming_attn_backend_requested"] = resolved_attn_backend
        backend_resolution["streaming_attn_backend_actual"] = resolved_attn_backend
        backend_resolution["streaming_attn_fallback_reason"] = None
        backend_resolution["rope_backend_actual"] = "torch_transformers"
        backend_resolution["rmsnorm_backend_actual"] = "torch_transformers"
        backend_resolution["result_interpretation_category"] = (
            "duo_streaming_like" if self.duo_enabled else "streamingtom"
        )
        manifest = {
            "method_name": self.method_name,
            "method_family": self.method_name,
            "kernel_backend_path": dict(self.runtime_backend_info),
            "backend_resolution": backend_resolution,
            "prompt_prefill_policy": "single_prefill_per_question",
            "streaming_protocol": {
                "causal_cutoff_policy": "sampled_timestamps_strictly_before_end_time",
                "frame_ingest_policy": "one_sampled_frame_per_forward_pass",
                "shared_state_across_questions": True,
                "offline_full_video_prefill": False,
                "feature_cache_requires_schedule_equivalence": False,
            },
            "cache_semantics_label": "streamingtom_video_reencode",
            "retrieval_offload_mode": "none",
            "full_attn_impl": self.attn_impl_used,
        }
        if self._duo_stats is not None:
            manifest["duo_deploy_config"] = {
                "deploy_window_class": "fixed_override",
                "deploy_sink_size": int(self._duo_stats["sink_size"]),
                "deploy_recent_size": int(self._duo_stats["recent_size"]),
            }
            manifest["duo_actual_sparsity"] = float(self._duo_stats["actual_sparsity"])
        return manifest

    def reset(self, sample_metadata: dict[str, Any]) -> None:
        self.frames_ingested = 0
        self.ingested_timestamps_sec = []
        self.ingest_latencies_sec = []
        self._frames = deque(maxlen=self.max_frames_context)

        # Proactively drop previous video pipelines/caches at video boundary.
        if hasattr(self.model, "_video_pipelines"):
            for vid, pipeline in list(self.model._video_pipelines.items()):
                pipeline.streamingtom_core.clear_cache(vid)
            self.model._video_pipelines.clear()

        sample_id = str(sample_metadata.get("sample_id") or sample_metadata.get("video_id") or "streamingtom_video")
        self._streaming_video_id = sample_id
        self._streaming_is_start = True
        if self.clear_cuda_cache_on_reset and self.device.type == "cuda":
            torch.cuda.empty_cache()
        # Reset peak memory once per video so peak_memory_bytes captures the full
        # model + OQM vision context footprint, not just the per-question QA delta.
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def ingest_frame(self, frame: np.ndarray, timestamp_sec: float) -> dict[str, Any]:
        start = time.perf_counter()
        frame_np = np.asarray(frame)
        if frame_np.ndim != 3:
            raise ValueError(f"Expected frame shape [H, W, C], got {tuple(frame_np.shape)}")
        self._last_h, self._last_w = int(frame_np.shape[0]), int(frame_np.shape[1])

        # Native incremental StreamingTom path: ingest one frame batch at a time
        # instead of storing/re-preprocessing the full frame history per question.
        pixel_values = self.image_processor.preprocess([frame_np], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            self.model.generate_with_streamingtom_streaming(
                images=pixel_values,
                questions=[],
                video_id=self._streaming_video_id,
                is_start=self._streaming_is_start,
            )
        self._streaming_is_start = False

        self.frames_ingested += 1
        self.ingested_timestamps_sec.append(float(timestamp_sec))
        latency = time.perf_counter() - start
        self.ingest_latencies_sec.append(float(latency))
        return {
            "timestamp_sec": float(timestamp_sec),
            "ingest_latency_sec": float(latency),
            "frame_token_count": 196,
        }

    def ingest_features(self, feature_tensor: np.ndarray, timestamp_sec: float) -> dict[str, Any]:
        blank = np.zeros((self._last_h, self._last_w, 3), dtype=np.uint8)
        return self.ingest_frame(blank, timestamp_sec)

    def answer_question(self, question: str, metadata: dict[str, Any] | None = None) -> MethodAnswer:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.inference_mode():
            answers = self.model.generate_with_streamingtom_streaming(
                images=None,
                questions=[{"batch_idx": -1, "input_ids": input_ids}],
                video_id=self._streaming_video_id,
                is_start=self._streaming_is_start,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                modalities=["video"],
            )
        self._streaming_is_start = False
        answer_latency_sec = time.perf_counter() - start

        ttft_sec = None
        retrieval_latency_sec = None
        try:
            from streamingtom.modules.streamingtom_context import StreamingTOMContext
            ctx = StreamingTOMContext.get_instance()
            ttft_sec = getattr(ctx, "last_ttft_sec", None)
            if ttft_sec is not None:
                ttft_sec = float(ttft_sec)
            retrieval_latency_sec = getattr(ctx, "last_retrieval_latency_sec", None)
            if retrieval_latency_sec is not None:
                retrieval_latency_sec = float(retrieval_latency_sec)
        except Exception:
            ttft_sec = None
            retrieval_latency_sec = None

        raw_answer = answers[0]["answer"] if answers else ""
        generated_token_ids: list[int]
        if isinstance(raw_answer, str):
            prediction = raw_answer.strip()
            generated_token_ids = [int(token_id) for token_id in self.tokenizer.encode(prediction, add_special_tokens=False)]
        elif isinstance(raw_answer, torch.Tensor):
            token_tensor = raw_answer[0].detach().cpu()
            generated_token_ids = [int(token_id) for token_id in token_tensor.tolist()]
            prediction = self.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            ).strip()
        else:
            prediction = str(raw_answer).strip()
            generated_token_ids = [int(token_id) for token_id in self.tokenizer.encode(prediction, add_special_tokens=False)]

        first_generated_token_id = int(generated_token_ids[0]) if generated_token_ids else None
        first_generated_token_text = (
            self.tokenizer.decode(
                [first_generated_token_id],
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if first_generated_token_id is not None
            else None
        )

        if self.device.type == "cuda":
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(self.device))
            current_memory_bytes = int(torch.cuda.memory_allocated(self.device))
        else:
            peak_memory_bytes = None
            current_memory_bytes = None

        stop_token_ids: list[int] = []
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_token_ids = [int(eos_token_id)]
        stop_token_ids = sorted(set(stop_token_ids))

        stop_reason = "max_new_tokens"
        stopped_on_token_id = None
        if generated_token_ids and generated_token_ids[-1] in stop_token_ids:
            stop_reason = "stop_token"
            stopped_on_token_id = int(generated_token_ids[-1])

        # Number of OQM groups retrieved per question (budget / group_size).
        _oqm_max = int(os.environ.get("OQM_RETRIEVAL_MAX_TOKENS", "12544"))
        _oqm_gs = int(os.environ.get("OQM_GROUP_SIZE", "50"))
        avg_retrieved_block_count = float(_oqm_max // _oqm_gs) if _oqm_gs > 0 else None

        stats = {
            "method_name": self.method_name,
            "ttft_sec": ttft_sec,
            "answer_latency_sec": float(answer_latency_sec),
            "current_memory_bytes": current_memory_bytes,
            "peak_memory_bytes": peak_memory_bytes,
            "cpu_offload_bytes_current": None,
            "cpu_offload_bytes_peak": None,
            "retrieval_latency_sec": retrieval_latency_sec,
            "avg_retrieved_block_count": avg_retrieved_block_count,
            "frames_ingested_so_far": int(self.frames_ingested),
            "generated_token_count": int(len(generated_token_ids)),
            "first_generated_token_id": first_generated_token_id,
            "first_generated_token_text": first_generated_token_text,
            "prompt_token_count": int(input_ids.shape[-1]),
            "prompt_prefill_mode": "inputs_embeds",
            "stop_reason": stop_reason,
            "stop_token_ids": stop_token_ids,
            "stopped_on_token_id": stopped_on_token_id,
        }
        if self._duo_stats is not None:
            stats["actual_sparsity"] = float(self._duo_stats["actual_sparsity"])
            stats["duo_sink_size"] = int(self._duo_stats["sink_size"])
            stats["duo_recent_size"] = int(self._duo_stats["recent_size"])
        return MethodAnswer(prediction=prediction, stats=stats)

    def get_runtime_stats(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "frames_ingested": int(self.frames_ingested),
            "avg_frame_ingest_latency_sec": (
                float(sum(self.ingest_latencies_sec) / len(self.ingest_latencies_sec))
                if self.ingest_latencies_sec
                else None
            ),
            "cumulative_frame_ingest_latency_sec": float(sum(self.ingest_latencies_sec)),
            "last_ingested_timestamp_sec": (
                float(self.ingested_timestamps_sec[-1]) if self.ingested_timestamps_sec else None
            ),
        }


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "annotation_path": args.annotation_path,
        "video_root": args.video_root,
        "hf_repo_id": args.hf_repo_id,
        "allow_hf_video_download": args.allow_hf_video_download,
        "model": args.model,
        "method": args.method,
        "sample_fps": args.sample_fps,
        "num_chunks": args.num_chunks,
        "chunk_index": args.chunk_index,
        "max_videos": args.max_videos,
        "video_offset": args.video_offset,
        "video_index": args.video_index,
        "video_id": args.video_id,
        "subsample_name": args.subsample_name,
        "max_conversations_per_video": args.max_conversations_per_video,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "device": args.device,
        "video_decode_threads": args.video_decode_threads,
        "clear_cuda_cache_on_reset": args.clear_cuda_cache_on_reset,
        "dtype": args.dtype,
        "attn_dir": args.duo_attn_dir,
        "sparsity": args.duo_sparsity,
        "threshold": args.duo_threshold,
        "deploy_sink_size": args.duo_sink_size,
        "deploy_recent_size": args.duo_recent_size,
        "duo_strict_no_sdpa_fallback": False,
        "n_local": None,
        "retrieve_size": None,
        "retrieve_chunk_size": None,
        "n_frame_tokens": 196,
        "rekv_fattn": False,
        "disable_rekv_pin_memory": False,
        "feature_cache_root": None,
        "ingest_source": "raw_frames",
    }


def build_method_from_args(args: argparse.Namespace) -> StreamingTomMethod:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    kwargs = {
        "pretrained": args.model,
        "method_name": args.method,
        "device": device,
        "dtype": dtype,
        "max_new_tokens": args.max_new_tokens,
        "streamingtom_root": args.streamingtom_root,
        "clear_cuda_cache_on_reset": bool(args.clear_cuda_cache_on_reset),
    }
    if args.method == "duo_plus_streamingtom":
        kwargs.update(
            {
                "duo_enabled": True,
                "duo_attn_dir": args.duo_attn_dir,
                "duo_heads_file": args.duo_heads_file,
                "duo_threshold": args.duo_threshold,
                "duo_sparsity": args.duo_sparsity,
                "duo_sink_size": args.duo_sink_size,
                "duo_recent_size": args.duo_recent_size,
            }
        )
    return StreamingTomMethod(**kwargs)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    run_config = build_run_config(args)

    if int(args.num_chunks) <= 0:
        raise ValueError("num_chunks must be >= 1")
    if not (0 <= int(args.chunk_index) < int(args.num_chunks)):
        raise ValueError("chunk_index must satisfy 0 <= chunk_index < num_chunks")

    dataset = build_dataset_from_args(args)
    samples = dataset.load(
        video_id=args.video_id,
        video_index=args.video_index,
        video_offset=args.video_offset,
        max_videos=args.max_videos,
    )
    if args.num_chunks > 1:
        chunk_size = (len(samples) + args.num_chunks - 1) // args.num_chunks
        chunk_start = args.chunk_index * chunk_size
        chunk_end = min(len(samples), chunk_start + chunk_size)
        samples = samples[chunk_start:chunk_end]
    if not samples:
        raise ValueError("No samples matched the requested dataset/video filters.")

    if args.max_conversations_per_video is not None:
        samples = [
            type(sample)(
                sample_id=sample.sample_id,
                video_id=sample.video_id,
                video_path=sample.video_path,
                duration=sample.duration,
                conversations=sample.conversations[: args.max_conversations_per_video],
                extra_metadata=sample.extra_metadata,
            )
            for sample in samples
        ]

    output_path = Path(args.output_path) if args.output_path else default_output_path(args)

    existing_videos: list[dict[str, Any]] = []
    existing_in_progress_video: dict[str, Any] | None = None
    started_at_utc = datetime.now(timezone.utc).isoformat()

    method = build_method_from_args(args)
    evaluation_manifest = build_evaluation_manifest(
        run_config=run_config,
        method_manifest=method.get_evaluation_manifest(),
        feature_cache_manifest=None,
    )

    if output_path.exists() and not args.resume and not args.overwrite_output:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --resume to continue or --overwrite-output."
        )

    if output_path.exists() and args.resume:
        with open(output_path, "r", encoding="utf-8") as handle:
            existing_payload = json.load(handle)
        validate_resume_payload(existing_payload, run_config, evaluation_manifest)
        existing_videos = existing_payload.get("videos", [])
        existing_in_progress_video = existing_payload.get("in_progress_video")
        started_at_utc = existing_payload.get("run_state", {}).get("started_at_utc") or started_at_utc

    payload = evaluate_samples(
        samples=samples,
        method=method,
        sample_fps=args.sample_fps,
        run_config=run_config,
        evaluation_manifest=evaluation_manifest,
        existing_videos=existing_videos,
        total_requested_videos=len(samples),
        started_at_utc=started_at_utc,
        checkpoint_path=output_path,
        flush_every_videos=max(1, int(args.flush_every_videos)),
        flush_every_conversations=max(1, int(args.flush_every_conversations)),
        show_progress_bar=not args.disable_progress_bar,
        feature_cache_root=None,
        existing_in_progress_video=existing_in_progress_video,
    )
    return normalize_result_payload_schema(payload)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    if args.resume and args.output_path is None:
        raise ValueError("--resume requires an explicit --output-path.")
    if args.flush_every_videos <= 0:
        raise ValueError("--flush-every-videos must be >= 1.")
    if args.flush_every_conversations <= 0:
        raise ValueError("--flush-every-conversations must be >= 1.")
    payload = run_eval(args)
    output_path = Path(args.output_path) if args.output_path else default_output_path(args)
    write_json_atomic(payload, output_path)
    print(f"Saved results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
