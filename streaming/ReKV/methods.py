from __future__ import annotations

import json
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.patch import enable_duo_attention_eval, load_full_attention_heads
from duo_attn.patch.attn_compat import FLASH_ATTN_AVAILABLE
from duo_attn.patch.flashinfer_utils import flashinfer
from duo_attn.patch.streaming_attn import is_blocksparse_available

from .rekv_core.patch import patch_hf
from .rekv_core.attention.dot_production_attention import (
    get_multi_stage_dot_production_attention,
)


DEFAULT_DUO_ATTN_DIR = (
    "outputs/train/"
    "0p5b_sink512_recent1024_maxlen32000_frames64_depth0p1-0p8_needles5_20260328_170632"
)
DEFAULT_INIT_PROMPT = (
    "<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user "
)


@dataclass(frozen=True)
class MethodAnswer:
    prediction: str
    stats: dict[str, Any]


@dataclass(frozen=True)
class AnswerPrefillPlan:
    past_key_values: Any
    prompt_text: str
    prefilled_logits: torch.Tensor | None = None
    prompt_token_count: int | None = None
    prompt_prefill_mode: str = "inputs_embeds"
    extra_stats: dict[str, Any] = field(default_factory=dict)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def ensure_generation_pad_token(
    model: LlavaOnevisionForConditionalGeneration,
    processor: Any,
) -> None:
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


def normalize_attn_dir(attn_dir: str) -> Path:
    attn_path = Path(attn_dir).expanduser()
    if attn_path.is_file():
        attn_path = attn_path.parent
    attn_path = attn_path.resolve(strict=False)
    if not attn_path.is_dir():
        raise FileNotFoundError(f"Attention directory not found: {attn_path}")
    return attn_path


def sparsify_attention_heads(
    full_attention_heads: torch.Tensor,
    *,
    seed: int,
    threshold: float | None = None,
    sparsity: float | None = None,
) -> tuple[torch.Tensor, float, float]:
    noisy_heads = full_attention_heads.detach().cpu().numpy().astype(np.float64)
    rng = random.Random(seed)
    for row_idx in range(noisy_heads.shape[0]):
        for col_idx in range(noisy_heads.shape[1]):
            noisy_heads[row_idx, col_idx] += rng.uniform(0.0, 1e-6)

    flat = noisy_heads.reshape(-1)
    if sparsity is not None:
        threshold = float(np.quantile(flat, sparsity))
        if sparsity >= 1:
            threshold = 2.0
        if sparsity <= 0:
            threshold = -1.0
    elif threshold is None:
        raise ValueError("Either threshold or sparsity must be provided.")

    binary = (noisy_heads >= threshold).astype(np.float32)
    actual_sparsity = 1.0 - float(binary.mean())
    return torch.from_numpy(binary), actual_sparsity, float(threshold)


def load_duo_attention_spec(
    attn_dir: str,
    *,
    seed: int,
    threshold: float | None,
    sparsity: float | None,
    deploy_sink_size: int | None,
    deploy_recent_size: int | None,
) -> tuple[torch.Tensor, int, int, dict[str, Any]]:
    normalized_dir = normalize_attn_dir(attn_dir)
    config_path = normalized_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing DuoAttention config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    full_attention_heads = load_full_attention_heads(str(normalized_dir))
    full_attention_heads, actual_sparsity, learned_threshold = sparsify_attention_heads(
        full_attention_heads,
        seed=seed,
        threshold=threshold,
        sparsity=sparsity,
    )
    sink_size = int(
        deploy_sink_size
        if deploy_sink_size is not None
        else config.get("deploy_sink_size") or config["sink_size"]
    )
    recent_size = int(
        deploy_recent_size
        if deploy_recent_size is not None
        else config.get("deploy_recent_size") or config["recent_size"]
    )
    config = dict(config)
    config["actual_sparsity"] = actual_sparsity
    config["learned_threshold"] = learned_threshold
    config["attn_dir"] = str(normalized_dir)
    return full_attention_heads, sink_size, recent_size, config


def _classify_duo_deploy_window(
    *,
    trained_sink_size: int,
    trained_recent_size: int,
    deploy_sink_size: int,
    deploy_recent_size: int,
) -> str:
    if (
        deploy_sink_size == trained_sink_size
        and deploy_recent_size == trained_recent_size
    ):
        return "training_aligned"
    if (
        deploy_sink_size <= trained_sink_size
        and deploy_recent_size <= trained_recent_size
    ):
        return "reduced_deploy_window"
    return "custom_deploy_window"


def collect_runtime_backend_info(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    device_name = None
    rocm_arch = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(device)
        except Exception:
            device_name = None
        try:
            device_properties = torch.cuda.get_device_properties(device)
            rocm_arch = getattr(device_properties, "gcnArchName", None)
            if not device_name:
                device_name = rocm_arch
        except Exception:
            device_properties = None

    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device_type": device.type,
        "device": str(device),
        "device_name": device_name,
        "rocm_arch": rocm_arch,
        "torch_dtype": str(dtype).replace("torch.", ""),
        "rocm_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "flash_attn_available": bool(FLASH_ATTN_AVAILABLE),
        "block_sparse_attn_available": bool(is_blocksparse_available()),
        "flashinfer_available": bool(flashinfer is not None),
        "attention_module_load_path": (
            "flash_attn" if FLASH_ATTN_AVAILABLE else "sdpa_fallback"
        ),
    }


def get_stop_token_ids(model: LlavaOnevisionForConditionalGeneration, tokenizer) -> set[int]:
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    else:
        eos_token_id = model.generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            stop_ids.update(int(value) for value in eos_token_id if value is not None)
        elif eos_token_id is not None:
            stop_ids.add(int(eos_token_id))
    return stop_ids


def build_init_prompt_ids(tokenizer, device: torch.device) -> torch.Tensor:
    return tokenizer(DEFAULT_INIT_PROMPT, return_tensors="pt").input_ids.to(device)


def build_question_prompt(question: str) -> str:
    return f"\n{question}<|im_end|><|im_start|>assistant\n"


def extract_video_features(
    model: LlavaOnevisionForConditionalGeneration,
    processor: Any,
    video_chunk: np.ndarray,
) -> torch.Tensor:
    video_chunk = np.asarray(video_chunk)
    if video_chunk.ndim == 3:
        video_chunk = np.expand_dims(video_chunk, axis=0)
    if video_chunk.ndim != 4:
        raise ValueError(
            "Expected video_chunk with shape [frames, height, width, channels], "
            f"got {tuple(video_chunk.shape)}"
        )

    pixel_values_videos = processor.video_processor(
        video_chunk,
        return_tensors="pt",
    ).pixel_values_videos
    return _project_pixel_values_videos(model, pixel_values_videos)


def _project_pixel_values_videos(
    model: LlavaOnevisionForConditionalGeneration,
    pixel_values_videos: torch.Tensor,
) -> torch.Tensor:
    lm_device = next(model.language_model.parameters()).device
    lm_dtype = next(model.language_model.parameters()).dtype
    vision_feature_layer = model.config.vision_feature_layer
    vision_feature_select_strategy = model.config.vision_feature_select_strategy

    if hasattr(model, "get_video_features"):
        vision_device = next(model.vision_tower.parameters()).device
        vision_dtype = next(model.vision_tower.parameters()).dtype
        pixel_values_videos = pixel_values_videos.to(device=vision_device, dtype=vision_dtype)
        video_features = model.get_video_features(
            pixel_values_videos,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        return video_features.to(device=lm_device, dtype=lm_dtype)

    vision_device = next(model.vision_tower.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype
    pixel_values_videos = pixel_values_videos.to(device=vision_device, dtype=vision_dtype)
    batch_size, frames, channels, height, width = pixel_values_videos.shape
    pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)

    if vision_feature_layer == -1:
        video_outputs = model.vision_tower(
            pixel_values_videos,
            output_hidden_states=False,
        )
        selected_video_feature = video_outputs.last_hidden_state
    else:
        video_outputs = model.vision_tower(
            pixel_values_videos,
            output_hidden_states=True,
        )
        selected_video_feature = video_outputs.hidden_states[vision_feature_layer]

    if vision_feature_select_strategy == "default":
        selected_video_feature = selected_video_feature[:, 1:]
    elif vision_feature_select_strategy != "full":
        raise ValueError(
            "Unsupported vision_feature_select_strategy="
            f"{vision_feature_select_strategy}"
        )

    video_features = model.multi_modal_projector(selected_video_feature)
    video_features = model.apply_pooling(video_features)
    video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)
    return video_features.to(device=lm_device, dtype=lm_dtype)


def extract_frame_features_batch(
    model: LlavaOnevisionForConditionalGeneration,
    processor: Any,
    frames: np.ndarray,
) -> torch.Tensor:
    frame_batch = np.asarray(frames)
    if frame_batch.ndim == 3:
        frame_batch = np.expand_dims(frame_batch, axis=0)
    if frame_batch.ndim != 4:
        raise ValueError(
            "Expected frames with shape [num_frames, height, width, channels], "
            f"got {tuple(frame_batch.shape)}"
        )

    feature_list: list[torch.Tensor] = []
    for frame in frame_batch:
        feature_list.append(extract_video_features(model, processor, frame)[0])

    if not feature_list:
        raise ValueError("extract_frame_features_batch requires at least one frame.")

    return torch.stack(feature_list, dim=0)


def extract_single_frame_features(
    model: LlavaOnevisionForConditionalGeneration,
    processor: Any,
    frame: np.ndarray,
) -> torch.Tensor:
    return extract_frame_features_batch(model, processor, np.expand_dims(frame, axis=0))[0]


def maybe_reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def read_peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def read_current_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.memory_allocated(device))


def _empty_retrieval_stats() -> dict[str, Any]:
    return {
        "retrieval_latency_sec": None,
        "retrieval_query_mode": None,
        "per_layer_retrieved_block_counts": [],
        "avg_retrieved_block_count": 0.0,
        "avg_global_block_count": 0.0,
        "per_layer_forced_local_token_counts": [],
        "avg_forced_local_token_count": 0.0,
        "per_layer_assembled_context_token_counts": [],
        "avg_assembled_context_token_count": 0.0,
        "per_layer_init_token_counts": [],
        "avg_init_token_count": 0.0,
        "per_layer_retrieved_old_token_counts": [],
        "avg_retrieved_old_token_count": 0.0,
        "local_window_forced": False,
        "retrieved_block_indices_union": [],
        "retrieved_frame_indices_union": [],
        "retrieved_timestamps_sec_union": [],
    }


def extract_logits_from_output(output, output_projection=None) -> torch.Tensor:
    if getattr(output, "logits", None) is not None:
        return output.logits
    if output_projection is None:
        raise AttributeError(
            "Language model output does not expose logits and no output projection was provided."
        )
    # Some ROCm/compiled inference paths return inference tensors here; clone so
    # the LM head can consume them without autograd bookkeeping errors.
    hidden_states = output.last_hidden_state.clone()
    return output_projection(hidden_states)


def greedy_decode_with_cache(
    *,
    language_model,
    output_projection=None,
    tokenizer,
    prompt_text: str,
    past_key_values,
    stop_token_ids: set[int],
    max_new_tokens: int,
    device: torch.device,
    prefilled_logits: torch.Tensor | None = None,
    prompt_token_count: int | None = None,
    prompt_prefill_mode: str | None = None,
) -> tuple[str, dict[str, Any]]:
    decode_start = time.perf_counter()
    if prefilled_logits is not None and prompt_text:
        raise ValueError("Provide either prompt_text or prefilled_logits, not both.")
    if prefilled_logits is None and not prompt_text:
        raise ValueError("greedy_decode_with_cache requires prompt_text or prefilled_logits.")

    actual_prompt_token_count = int(prompt_token_count or 0)
    if max_new_tokens <= 0:
        return "", {
            "ttft_sec": 0.0,
            "answer_latency_sec": 0.0,
            "generated_token_count": 0,
            "prompt_token_count": actual_prompt_token_count,
            "prompt_prefill_mode": prompt_prefill_mode or "inputs_embeds",
        }

    output_ids: list[int] = []
    stop_reason = "max_new_tokens"

    with torch.inference_mode():
        if prefilled_logits is None:
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            prompt_embeds = language_model.get_input_embeddings()(prompt_ids)
            out = language_model(
                inputs_embeds=prompt_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            working_cache = out.past_key_values
            logits = extract_logits_from_output(out, output_projection)
            actual_prompt_token_count = int(prompt_ids.shape[-1])
            actual_prompt_prefill_mode = prompt_prefill_mode or "inputs_embeds"
        else:
            working_cache = past_key_values
            logits = prefilled_logits
            actual_prompt_prefill_mode = prompt_prefill_mode or "external_prefill"

        next_token = int(torch.argmax(logits[0, -1, :]).item())
        output_ids.append(next_token)
        ttft_sec = time.perf_counter() - decode_start
        if next_token in stop_token_ids:
            stop_reason = "stop_token"

        for _ in range(1, max_new_tokens):
            if next_token in stop_token_ids:
                break
            out = language_model(
                input_ids=torch.tensor([[next_token]], device=device),
                use_cache=True,
                past_key_values=working_cache,
            )
            logits = extract_logits_from_output(out, output_projection)
            working_cache = out.past_key_values
            next_token = int(torch.argmax(logits[0, -1, :]).item())
            output_ids.append(next_token)
            if next_token in stop_token_ids:
                stop_reason = "stop_token"

    prediction = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True,
    ).strip()
    return prediction, {
        "ttft_sec": ttft_sec,
        "answer_latency_sec": time.perf_counter() - decode_start,
        "generated_token_count": len(output_ids),
        "prompt_token_count": actual_prompt_token_count,
        "prompt_prefill_mode": actual_prompt_prefill_mode,
        "first_generated_token_id": int(output_ids[0]) if output_ids else None,
        "first_generated_token_text": (
            tokenizer.decode(
                [output_ids[0]],
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if output_ids
            else ""
        ),
        "stop_token_ids": sorted(int(token_id) for token_id in stop_token_ids),
        "stopped_on_token_id": int(output_ids[-1]) if output_ids[-1] in stop_token_ids else None,
        "stop_reason": stop_reason,
    }


class StreamingMethod(ABC):
    method_name: str

    @abstractmethod
    def reset(self, sample_metadata: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def ingest_frame(self, frame: np.ndarray, timestamp_sec: float) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def ingest_features(self, feature_tensor: torch.Tensor, timestamp_sec: float) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def answer_question(self, question: str, metadata: dict[str, Any] | None = None) -> MethodAnswer:
        raise NotImplementedError

    @abstractmethod
    def get_runtime_stats(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_evaluation_manifest(self) -> dict[str, Any]:
        return {}


class _BaseLlavaStreamingMethod(StreamingMethod):
    def __init__(
        self,
        *,
        pretrained: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        clear_cuda_cache_on_reset: bool = False,
    ) -> None:
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.max_new_tokens = int(max_new_tokens)
        self.clear_cuda_cache_on_reset = bool(clear_cuda_cache_on_reset)
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()
        ensure_generation_pad_token(self.model, self.processor)
        self.tokenizer = self.processor.tokenizer
        self.stop_token_ids = get_stop_token_ids(self.model, self.tokenizer)
        self.init_prompt_ids = build_init_prompt_ids(self.tokenizer, self.device)
        self.base_cache = None
        self.sample_metadata: dict[str, Any] = {}
        self.frames_ingested = 0
        self.ingested_timestamps_sec: list[float] = []
        self.ingest_latencies_sec: list[float] = []
        self.runtime_backend_info = collect_runtime_backend_info(self.device, self.dtype)

    def _language_model_io_spec(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.model.language_model.parameters())
        return parameter.device, parameter.dtype

    def extract_frame_features_batch(self, frames: np.ndarray) -> torch.Tensor:
        return extract_frame_features_batch(self.model, self.processor, frames)

    def extract_single_frame_features(self, frame: np.ndarray) -> torch.Tensor:
        return extract_single_frame_features(self.model, self.processor, frame)

    def _encode_init_prompt(self) -> None:
        with torch.inference_mode():
            out = self.model.language_model(
                input_ids=self.init_prompt_ids,
                use_cache=True,
                return_dict=True,
            )
        self.base_cache = out.past_key_values

    def reset(self, sample_metadata: dict[str, Any]) -> None:
        self.sample_metadata = dict(sample_metadata)
        self.frames_ingested = 0
        self.ingested_timestamps_sec = []
        self.ingest_latencies_sec = []
        self.base_cache = None
        if self.device.type == "cuda" and self.clear_cuda_cache_on_reset:
            torch.cuda.empty_cache()
        self._encode_init_prompt()

    def ingest_frame(self, frame: np.ndarray, timestamp_sec: float) -> dict[str, Any]:
        frame_features = self.extract_single_frame_features(frame)
        return self.ingest_features(frame_features, timestamp_sec)

    def ingest_features(self, feature_tensor: torch.Tensor, timestamp_sec: float) -> dict[str, Any]:
        if not isinstance(feature_tensor, torch.Tensor):
            raise TypeError(
                "feature_tensor must be a torch.Tensor, "
                f"got {type(feature_tensor).__name__}"
            )

        video_features = feature_tensor
        if video_features.ndim == 2:
            video_features = video_features.unsqueeze(0)
        if video_features.ndim != 3:
            raise ValueError(
                "feature_tensor must have shape [tokens, hidden] or [1, tokens, hidden], "
                f"got {tuple(video_features.shape)}"
            )

        lm_device, lm_dtype = self._language_model_io_spec()
        start = time.perf_counter()
        with torch.inference_mode():
            video_features = video_features.to(device=lm_device, dtype=lm_dtype)
            self._validate_video_features(video_features)
            out = self.model.language_model(
                inputs_embeds=video_features,
                use_cache=True,
                past_key_values=self.base_cache,
            )
        self.base_cache = out.past_key_values
        ingest_latency_sec = time.perf_counter() - start
        self.frames_ingested += 1
        self.ingested_timestamps_sec.append(float(timestamp_sec))
        self.ingest_latencies_sec.append(ingest_latency_sec)
        return {
            "timestamp_sec": float(timestamp_sec),
            "ingest_latency_sec": ingest_latency_sec,
            "frame_token_count": int(video_features.shape[1]),
        }

    def answer_question(self, question: str, metadata: dict[str, Any] | None = None) -> MethodAnswer:
        metadata = metadata or {}
        maybe_reset_peak_memory(self.device)
        prompt_text = build_question_prompt(question)
        prefill_plan = self._prepare_answer_prefill(prompt_text, metadata)
        prediction, decode_stats = greedy_decode_with_cache(
            language_model=self.model.language_model,
            output_projection=self.model.get_output_embeddings(),
            tokenizer=self.tokenizer,
            prompt_text=prefill_plan.prompt_text,
            past_key_values=prefill_plan.past_key_values,
            stop_token_ids=self.stop_token_ids,
            max_new_tokens=self.max_new_tokens,
            device=self.device,
            prefilled_logits=prefill_plan.prefilled_logits,
            prompt_token_count=prefill_plan.prompt_token_count,
            prompt_prefill_mode=prefill_plan.prompt_prefill_mode,
        )
        current_memory_bytes = read_current_memory_bytes(self.device)
        peak_memory_bytes = read_peak_memory_bytes(self.device)
        method_stats = self._build_answer_stats(question, metadata)
        method_stats.update(prefill_plan.extra_stats)
        method_stats.update(decode_stats)
        method_stats["current_memory_bytes"] = current_memory_bytes
        method_stats["peak_memory_bytes"] = peak_memory_bytes
        method_stats["frames_ingested_so_far"] = self.frames_ingested
        return MethodAnswer(prediction=prediction, stats=method_stats)

    @abstractmethod
    def _validate_video_features(self, video_features: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def _prepare_answer_prefill(self, prompt_text: str, metadata: dict[str, Any]) -> AnswerPrefillPlan:
        raise NotImplementedError

    def _build_answer_stats(self, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"method_name": self.method_name}

    def get_runtime_stats(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "frames_ingested": self.frames_ingested,
            "avg_frame_ingest_latency_sec": (
                float(sum(self.ingest_latencies_sec) / len(self.ingest_latencies_sec))
                if self.ingest_latencies_sec
                else None
            ),
            "last_ingested_timestamp_sec": (
                self.ingested_timestamps_sec[-1] if self.ingested_timestamps_sec else None
            ),
        }

    def get_evaluation_manifest(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "method_family": self.method_name,
            "kernel_backend_path": dict(self.runtime_backend_info),
            "prompt_prefill_policy": "single_prefill_per_question",
            "streaming_protocol": {
                "causal_cutoff_policy": "sampled_timestamps_strictly_before_end_time",
                "frame_ingest_policy": "one_sampled_frame_per_forward_pass",
                "shared_state_across_questions": True,
                "offline_full_video_prefill": False,
                "feature_cache_requires_schedule_equivalence": True,
            },
        }


class DuoStreamingMethod(_BaseLlavaStreamingMethod):
    method_name = "duo_streaming"

    def __init__(
        self,
        *,
        pretrained: str,
        attn_dir: str = DEFAULT_DUO_ATTN_DIR,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        threshold: float | None = None,
        sparsity: float | None = 0.5,
        seed: int = 42,
        deploy_sink_size: int | None = None,
        deploy_recent_size: int | None = None,
        clear_cuda_cache_on_reset: bool = False,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            clear_cuda_cache_on_reset=clear_cuda_cache_on_reset,
        )
        (
            full_attention_heads,
            sink_size,
            recent_size,
            attn_config,
        ) = load_duo_attention_spec(
            attn_dir,
            seed=seed,
            threshold=threshold,
            sparsity=sparsity,
            deploy_sink_size=deploy_sink_size,
            deploy_recent_size=deploy_recent_size,
        )
        enable_duo_attention_eval(self.model, full_attention_heads, sink_size, recent_size)
        self.attn_dir = attn_config["attn_dir"]
        self.trained_sink_size = int(attn_config["sink_size"])
        self.trained_recent_size = int(attn_config["recent_size"])
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.actual_sparsity = float(attn_config["actual_sparsity"])
        self.learned_threshold = float(attn_config["learned_threshold"])
        self.duo_deploy_window_class = _classify_duo_deploy_window(
            trained_sink_size=self.trained_sink_size,
            trained_recent_size=self.trained_recent_size,
            deploy_sink_size=self.sink_size,
            deploy_recent_size=self.recent_size,
        )

    def _validate_video_features(self, video_features: torch.Tensor) -> None:
        if video_features.ndim != 3 or video_features.shape[0] != 1:
            raise ValueError(f"Unexpected video feature shape for DuoAttention: {tuple(video_features.shape)}")

    def _prepare_answer_prefill(self, prompt_text: str, metadata: dict[str, Any]) -> AnswerPrefillPlan:
        return AnswerPrefillPlan(
            past_key_values=self.base_cache,
            prompt_text=prompt_text,
        )

    def _build_answer_stats(self, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "attn_dir": self.attn_dir,
            "sink_size": self.sink_size,
            "recent_size": self.recent_size,
            "effective_sink_window_tokens": self.sink_size,
            "effective_recent_window_tokens": self.recent_size,
            "actual_sparsity": self.actual_sparsity,
            "learned_threshold": self.learned_threshold,
            "trained_sink_size": self.trained_sink_size,
            "trained_recent_size": self.trained_recent_size,
            "duo_deploy_window_class": self.duo_deploy_window_class,
            "duo_cache_semantics": "official_tuple_kv_compressed_streaming",
            "duo_eval_attention_backend": self.runtime_backend_info["attention_module_load_path"],
        }

    def get_evaluation_manifest(self) -> dict[str, Any]:
        manifest = super().get_evaluation_manifest()
        manifest.update(
            {
                "method_family": "duo_streaming",
                "cache_semantics_label": "duo_tuple_kv_compressed_streaming",
                "duo_attention_backend": self.runtime_backend_info["attention_module_load_path"],
                "duo_deploy_config": {
                    "trained_sink_size": self.trained_sink_size,
                    "trained_recent_size": self.trained_recent_size,
                    "deploy_sink_size": self.sink_size,
                    "deploy_recent_size": self.recent_size,
                    "deploy_window_class": self.duo_deploy_window_class,
                    "actual_sparsity": self.actual_sparsity,
                },
            }
        )
        return manifest


class FullStreamingMethod(_BaseLlavaStreamingMethod):
    method_name = "full_streaming"

    def _validate_video_features(self, video_features: torch.Tensor) -> None:
        if video_features.ndim != 3 or video_features.shape[0] != 1:
            raise ValueError(
                f"Unexpected video feature shape for full streaming: {tuple(video_features.shape)}"
            )

    def _prepare_answer_prefill(self, prompt_text: str, metadata: dict[str, Any]) -> AnswerPrefillPlan:
        return AnswerPrefillPlan(
            past_key_values=self.base_cache,
            prompt_text=prompt_text,
        )

    def _build_answer_stats(self, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"method_name": self.method_name}

    def get_evaluation_manifest(self) -> dict[str, Any]:
        manifest = super().get_evaluation_manifest()
        manifest.update(
            {
                "method_family": "full_streaming",
                "cache_semantics_label": "plain_full_streaming_cache",
                "retrieval_offload_mode": "none",
            }
        )
        return manifest


class ReKVStreamingMethod(_BaseLlavaStreamingMethod):
    method_name = "rekv"

    def __init__(
        self,
        *,
        pretrained: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        n_local: int = 15000,
        retrieve_size: int = 64,
        retrieve_chunk_size: int = 1,
        n_frame_tokens: int = 196,
        fattn: bool = False,
        pin_memory: bool = True,
        short_memory_only: bool = False,
        clear_cuda_cache_on_reset: bool = False,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            clear_cuda_cache_on_reset=clear_cuda_cache_on_reset,
        )
        self._init_rekv_backend(
            n_local=n_local,
            retrieve_size=retrieve_size,
            retrieve_chunk_size=retrieve_chunk_size,
            n_frame_tokens=n_frame_tokens,
            fattn=fattn,
            pin_memory=pin_memory,
            short_memory_only=short_memory_only,
        )

    def _init_rekv_backend(
        self,
        *,
        n_local: int,
        retrieve_size: int,
        retrieve_chunk_size: int,
        n_frame_tokens: int,
        fattn: bool,
        pin_memory: bool,
        short_memory_only: bool,
        n_init_override: int | None = None,
    ) -> None:
        self.n_local = int(n_local)
        self.retrieve_size = int(retrieve_size)
        self.retrieve_chunk_size = int(retrieve_chunk_size)
        self.n_frame_tokens = int(n_frame_tokens)
        self.fattn = bool(fattn)
        self.short_memory_only = bool(short_memory_only)
        self.pin_memory = bool(pin_memory and self.device.type == "cuda")
        _, actual_rekv_fattn = get_multi_stage_dot_production_attention(self.fattn)
        self.rekv_dot_backend_requested = "triton_flash_attn" if self.fattn else "torch"
        self.rekv_dot_backend_actual = (
            "triton_flash_attn" if actual_rekv_fattn else "torch"
        )
        effective_n_init = (
            int(n_init_override) if n_init_override is not None
            else int(self.init_prompt_ids.shape[-1])
        )
        self.model.language_model = patch_hf(
            self.model.language_model,
            n_init=effective_n_init,
            n_local=self.n_local,
            fattn=self.fattn,
            block_size=self.n_frame_tokens,
            topk=self.retrieve_size,
            chunk_size=self.retrieve_chunk_size,
            max_cached_block=128,
            exc_block_size=self.n_frame_tokens,
            pin_memory=self.pin_memory,
            short_memory_only=self.short_memory_only,
        )
        self.last_retrieval_stats: dict[str, Any] = _empty_retrieval_stats()
        self.cpu_offload_bytes_peak = 0

    def reset(self, sample_metadata: dict[str, Any]) -> None:
        super().reset(sample_metadata)
        self.last_retrieval_stats = _empty_retrieval_stats()
        self.cpu_offload_bytes_peak = 0

    def _current_cpu_offload_bytes(self) -> int:
        if self.base_cache is None:
            return 0
        total = 0
        for layer_kv in self.base_cache:
            if hasattr(layer_kv, "calculate_cpu_memory"):
                total += int(layer_kv.calculate_cpu_memory())
        return total

    def _update_cpu_offload_peak(self) -> int:
        current_bytes = self._current_cpu_offload_bytes()
        self.cpu_offload_bytes_peak = max(self.cpu_offload_bytes_peak, current_bytes)
        return current_bytes

    def ingest_features(self, feature_tensor: torch.Tensor, timestamp_sec: float) -> dict[str, Any]:
        return super().ingest_features(feature_tensor, timestamp_sec)

    def _retrieved_timestamps(self, block_indices: list[int]) -> list[float]:
        timestamps: list[float] = []
        for block_idx in block_indices:
            if 0 <= block_idx < len(self.ingested_timestamps_sec):
                timestamps.append(float(self.ingested_timestamps_sec[block_idx]))
        return timestamps

    def _validate_video_features(self, video_features: torch.Tensor) -> None:
        if video_features.ndim != 3 or video_features.shape[0] != 1:
            raise ValueError(f"Unexpected video feature shape for ReKV: {tuple(video_features.shape)}")
        if video_features.shape[1] != self.n_frame_tokens:
            raise ValueError(
                "ReKV expects one ingested frame to map to exactly "
                f"{self.n_frame_tokens} visual tokens, got {video_features.shape[1]}."
            )

    def _prepare_answer_prefill(self, prompt_text: str, metadata: dict[str, Any]) -> AnswerPrefillPlan:
        if self.short_memory_only:
            self.last_retrieval_stats = _empty_retrieval_stats()
            return AnswerPrefillPlan(
                past_key_values=self.base_cache,
                prompt_text=prompt_text,
            )

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
        retrieval_start = time.perf_counter()
        for layer_kv in self.base_cache:
            layer_kv.set_retrieval()

        with torch.inference_mode():
            out = self.model.language_model(
                input_ids=prompt_ids,
                use_cache=True,
                past_key_values=self.base_cache,
            )
        retrieval_past = out.past_key_values
        retrieval_latency_sec = time.perf_counter() - retrieval_start
        prefilled_logits = extract_logits_from_output(
            out,
            self.model.get_output_embeddings(),
        )

        per_layer_retrieved_counts: list[int] = []
        per_layer_global_blocks: list[int] = []
        per_layer_forced_local_token_counts: list[int] = []
        per_layer_assembled_context_token_counts: list[int] = []
        per_layer_init_token_counts: list[int] = []
        per_layer_retrieved_old_token_counts: list[int] = []
        retrieved_block_indices_union: set[int] = set()
        for layer_kv in self.base_cache:
            retrieved_indices = layer_kv.retrieved_block_indices or []
            if retrieved_indices and retrieved_indices[0] is not None:
                flat_indices = sorted(
                    {
                        int(block_idx)
                        for batch_indices in retrieved_indices
                        for block_idx in (batch_indices or [])
                        if block_idx is not None
                    }
                )
                per_layer_retrieved_counts.append(len(flat_indices))
                retrieved_block_indices_union.update(flat_indices)
            else:
                per_layer_retrieved_counts.append(0)
            per_layer_global_blocks.append(int(getattr(layer_kv, "num_global_block", 0)))
            answer_context_stats = dict(getattr(layer_kv, "last_answer_context_stats", {}) or {})
            per_layer_forced_local_token_counts.append(
                int(answer_context_stats.get("forced_local_token_count", 0))
            )
            per_layer_assembled_context_token_counts.append(
                int(answer_context_stats.get("assembled_context_token_count", 0))
            )
            per_layer_init_token_counts.append(
                int(answer_context_stats.get("init_token_count", 0))
            )
            per_layer_retrieved_old_token_counts.append(
                int(answer_context_stats.get("retrieved_old_token_count", 0))
            )
            layer_kv.reset_retrieval()

        retrieved_block_indices_sorted = sorted(retrieved_block_indices_union)
        self.last_retrieval_stats = {
            "retrieval_latency_sec": retrieval_latency_sec,
            "retrieval_query_mode": "full_prompt_prefill",
            "per_layer_retrieved_block_counts": per_layer_retrieved_counts,
            "avg_retrieved_block_count": (
                float(sum(per_layer_retrieved_counts) / len(per_layer_retrieved_counts))
                if per_layer_retrieved_counts
                else 0.0
            ),
            "avg_global_block_count": (
                float(sum(per_layer_global_blocks) / len(per_layer_global_blocks))
                if per_layer_global_blocks
                else 0.0
            ),
            "per_layer_forced_local_token_counts": per_layer_forced_local_token_counts,
            "avg_forced_local_token_count": (
                float(sum(per_layer_forced_local_token_counts) / len(per_layer_forced_local_token_counts))
                if per_layer_forced_local_token_counts
                else 0.0
            ),
            "per_layer_assembled_context_token_counts": per_layer_assembled_context_token_counts,
            "avg_assembled_context_token_count": (
                float(sum(per_layer_assembled_context_token_counts) / len(per_layer_assembled_context_token_counts))
                if per_layer_assembled_context_token_counts
                else 0.0
            ),
            "per_layer_init_token_counts": per_layer_init_token_counts,
            "avg_init_token_count": (
                float(sum(per_layer_init_token_counts) / len(per_layer_init_token_counts))
                if per_layer_init_token_counts
                else 0.0
            ),
            "per_layer_retrieved_old_token_counts": per_layer_retrieved_old_token_counts,
            "avg_retrieved_old_token_count": (
                float(sum(per_layer_retrieved_old_token_counts) / len(per_layer_retrieved_old_token_counts))
                if per_layer_retrieved_old_token_counts
                else 0.0
            ),
            "local_window_forced": any(count > 0 for count in per_layer_forced_local_token_counts),
            "retrieved_block_indices_union": retrieved_block_indices_sorted,
            "retrieved_frame_indices_union": list(retrieved_block_indices_sorted),
            "retrieved_timestamps_sec_union": self._retrieved_timestamps(
                retrieved_block_indices_sorted
            ),
        }
        return AnswerPrefillPlan(
            past_key_values=retrieval_past,
            prompt_text="",
            prefilled_logits=prefilled_logits,
            prompt_token_count=int(prompt_ids.shape[-1]),
            prompt_prefill_mode="retrieval_prefill",
        )

    def _build_answer_stats(self, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        current_offload_bytes = self._update_cpu_offload_peak()
        return {
            "method_name": self.method_name,
            "retrieval_policy": (
                "short_memory_only_discard_old_context"
                if self.short_memory_only
                else "rekv_native"
            ),
            "cpu_offload_bytes_current": current_offload_bytes,
            "cpu_offload_bytes_peak": self.cpu_offload_bytes_peak,
            "n_local": self.n_local,
            "retrieve_size": self.retrieve_size,
            "retrieve_chunk_size": self.retrieve_chunk_size,
            "n_frame_tokens": self.n_frame_tokens,
            "fattn": self.fattn,
            "rekv_dot_backend_requested": self.rekv_dot_backend_requested,
            "rekv_dot_backend_actual": self.rekv_dot_backend_actual,
            "retrieval_context_layout": "init_plus_retrieved_old_plus_forced_local",
            **self.last_retrieval_stats,
        }

    def get_runtime_stats(self) -> dict[str, Any]:
        runtime_stats = super().get_runtime_stats()
        current_offload_bytes = self._current_cpu_offload_bytes()
        self.cpu_offload_bytes_peak = max(self.cpu_offload_bytes_peak, current_offload_bytes)
        runtime_stats.update(
            {
                "cpu_offload_bytes_current": current_offload_bytes,
                "cpu_offload_bytes_peak": self.cpu_offload_bytes_peak,
            }
        )
        return runtime_stats

    def get_evaluation_manifest(self) -> dict[str, Any]:
        manifest = super().get_evaluation_manifest()
        manifest.update(
            {
                "method_family": self.method_name,
                "cache_semantics_label": (
                    "short_memory_only_discard_old_context"
                    if self.short_memory_only
                    else "rekv_init_retrieved_old_forced_local"
                ),
                "retrieval_offload_mode": (
                    "disabled_short_memory_only"
                    if self.short_memory_only
                    else "rekv_context_manager_with_cpu_offload"
                ),
                "rekv_config": {
                    "n_local": self.n_local,
                    "retrieve_size": self.retrieve_size,
                    "retrieve_chunk_size": self.retrieve_chunk_size,
                    "n_frame_tokens": self.n_frame_tokens,
                    "fattn_requested": self.fattn,
                    "dot_backend_requested": self.rekv_dot_backend_requested,
                    "dot_backend_actual": self.rekv_dot_backend_actual,
                },
            }
        )
        return manifest


class ReKVNoOffloadStreamingMethod(ReKVStreamingMethod):
    method_name = "rekv_no_offload"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, short_memory_only=True)


class DuoPlusReKVStreamingMethod(ReKVStreamingMethod):
    method_name = "duo_plus_rekv"

    def __init__(
        self,
        *,
        pretrained: str,
        attn_dir: str = DEFAULT_DUO_ATTN_DIR,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        threshold: float | None = None,
        sparsity: float | None = 0.5,
        seed: int = 42,
        deploy_sink_size: int | None = None,
        deploy_recent_size: int | None = None,
        n_local: int = 15000,
        retrieve_size: int = 64,
        retrieve_chunk_size: int = 1,
        n_frame_tokens: int = 196,
        fattn: bool = False,
        pin_memory: bool = True,
        clear_cuda_cache_on_reset: bool = False,
    ) -> None:
        _BaseLlavaStreamingMethod.__init__(
            self,
            pretrained=pretrained,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            clear_cuda_cache_on_reset=clear_cuda_cache_on_reset,
        )
        (
            full_attention_heads,
            sink_size,
            recent_size,
            attn_config,
        ) = load_duo_attention_spec(
            attn_dir,
            seed=seed,
            threshold=threshold,
            sparsity=sparsity,
            deploy_sink_size=deploy_sink_size,
            deploy_recent_size=deploy_recent_size,
        )
        # Only apply the weight reordering and full_attention_heads registration
        # from DuoAttention — skip the tuple KV cache patching since ReKV's
        # patch_hf will overwrite the model forward anyway.
        from duo_attn.patch.llava_onevision import (
            _enable_qwen2_layers_duo_attention_eval,
            _get_qwen2_layers,
        )
        _enable_qwen2_layers_duo_attention_eval(
            _get_qwen2_layers(self.model),
            full_attention_heads,
            sink_size,
            recent_size,
        )
        _lang = self.model.language_model
        _layers = _lang.layers if hasattr(_lang, "layers") else _lang.model.layers
        for layer in _layers:
            layer.self_attn.rekv_duo_enabled = True

        self.attn_dir = attn_config["attn_dir"]
        self.trained_sink_size = int(attn_config["sink_size"])
        self.trained_recent_size = int(attn_config["recent_size"])
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.actual_sparsity = float(attn_config["actual_sparsity"])
        self.learned_threshold = float(attn_config["learned_threshold"])
        self.duo_deploy_window_class = _classify_duo_deploy_window(
            trained_sink_size=self.trained_sink_size,
            trained_recent_size=self.trained_recent_size,
            deploy_sink_size=self.sink_size,
            deploy_recent_size=self.recent_size,
        )

        self.n_local = int(n_local)
        # Compute n_init for the ReKV context manager that preserves enough
        # early tokens to serve as attention sinks for DuoAttention's streaming
        # heads (trained with sink_size tokens).  n_init must be block-aligned
        # after the init prompt, otherwise the context manager's block
        # offloading assertion will fail.
        init_prompt_len = int(self.init_prompt_ids.shape[-1])
        if sink_size > init_prompt_len:
            extra_blocks = math.ceil(
                (sink_size - init_prompt_len) / n_frame_tokens
            )
            duo_n_init = init_prompt_len + extra_blocks * n_frame_tokens
        else:
            duo_n_init = init_prompt_len
        self.duo_n_init = duo_n_init
        self._init_rekv_backend(
            n_local=n_local,
            retrieve_size=retrieve_size,
            retrieve_chunk_size=retrieve_chunk_size,
            n_frame_tokens=n_frame_tokens,
            fattn=fattn,
            pin_memory=pin_memory,
            short_memory_only=False,
            n_init_override=duo_n_init,
        )

    def _build_answer_stats(self, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        current_offload_bytes = self._update_cpu_offload_peak()
        return {
            "method_name": self.method_name,
            "integration_mode": "rekv_memory_plus_duo_lm",
            "retrieval_policy": "rekv_native",
            "answer_attention_policy": "duo_post_retrieval",
            "cpu_offload_bytes_current": current_offload_bytes,
            "cpu_offload_bytes_peak": self.cpu_offload_bytes_peak,
            "attn_dir": self.attn_dir,
            "duo_n_init": self.duo_n_init,
            "sink_size": self.sink_size,
            "recent_size": self.recent_size,
            "actual_sparsity": self.actual_sparsity,
            "learned_threshold": self.learned_threshold,
            "trained_sink_size": self.trained_sink_size,
            "trained_recent_size": self.trained_recent_size,
            "duo_deploy_window_class": self.duo_deploy_window_class,
            "duo_cache_semantics": "hybrid_rekv_context_plus_duo_head_routing",
            "duo_eval_attention_backend": self.runtime_backend_info["attention_module_load_path"],
            "effective_duo_recent_window_tokens": min(self.n_local, self.recent_size),
            "n_local": self.n_local,
            "retrieve_size": self.retrieve_size,
            "retrieve_chunk_size": self.retrieve_chunk_size,
            "n_frame_tokens": self.n_frame_tokens,
            "fattn": self.fattn,
            "rekv_dot_backend_requested": self.rekv_dot_backend_requested,
            "rekv_dot_backend_actual": self.rekv_dot_backend_actual,
            "retrieval_context_layout": "init_plus_retrieved_old_plus_forced_local",
            **self.last_retrieval_stats,
        }

    def get_evaluation_manifest(self) -> dict[str, Any]:
        manifest = super().get_evaluation_manifest()
        manifest.update(
            {
                "method_family": "duo_plus_rekv",
                "cache_semantics_label": "hybrid_rekv_context_plus_duo_head_routing",
                "retrieval_offload_mode": "rekv_context_manager_with_cpu_offload",
                "hybrid_mode_label": "approximate_duo_over_rekv_context",
                "duo_attention_backend": self.runtime_backend_info["attention_module_load_path"],
                "duo_deploy_config": {
                    "trained_sink_size": self.trained_sink_size,
                    "trained_recent_size": self.trained_recent_size,
                    "deploy_sink_size": self.sink_size,
                    "deploy_recent_size": self.recent_size,
                    "deploy_window_class": self.duo_deploy_window_class,
                    "actual_sparsity": self.actual_sparsity,
                    "effective_recent_window_tokens": min(self.n_local, self.recent_size),
                },
            }
        )
        return manifest


def build_method_from_args(args) -> StreamingMethod:
    shared_kwargs = {
        "pretrained": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "clear_cuda_cache_on_reset": args.clear_cuda_cache_on_reset,
    }
    if args.method == "full_streaming":
        return FullStreamingMethod(**shared_kwargs)
    if args.method == "duo_streaming":
        return DuoStreamingMethod(
            **shared_kwargs,
            attn_dir=args.attn_dir or DEFAULT_DUO_ATTN_DIR,
            threshold=args.threshold,
            sparsity=args.sparsity,
            seed=args.seed,
            deploy_sink_size=args.deploy_sink_size,
            deploy_recent_size=args.deploy_recent_size,
        )
    if args.method == "rekv":
        return ReKVStreamingMethod(
            **shared_kwargs,
            n_local=args.n_local,
            retrieve_size=args.retrieve_size,
            retrieve_chunk_size=args.retrieve_chunk_size,
            n_frame_tokens=args.n_frame_tokens,
            fattn=args.rekv_fattn,
            pin_memory=not args.disable_rekv_pin_memory,
        )
    if args.method == "rekv_no_offload":
        return ReKVNoOffloadStreamingMethod(
            **shared_kwargs,
            n_local=args.n_local,
            retrieve_size=args.retrieve_size,
            retrieve_chunk_size=args.retrieve_chunk_size,
            n_frame_tokens=args.n_frame_tokens,
            fattn=args.rekv_fattn,
            pin_memory=not args.disable_rekv_pin_memory,
        )
    if args.method == "duo_plus_rekv":
        return DuoPlusReKVStreamingMethod(
            **shared_kwargs,
            attn_dir=args.attn_dir or DEFAULT_DUO_ATTN_DIR,
            threshold=args.threshold,
            sparsity=args.sparsity,
            seed=args.seed,
            deploy_sink_size=args.deploy_sink_size,
            deploy_recent_size=args.deploy_recent_size,
            n_local=args.n_local,
            retrieve_size=args.retrieve_size,
            retrieve_chunk_size=args.retrieve_chunk_size,
            n_frame_tokens=args.n_frame_tokens,
            fattn=args.rekv_fattn,
            pin_memory=not args.disable_rekv_pin_memory,
        )
    raise ValueError(f"Unsupported method: {args.method}")
