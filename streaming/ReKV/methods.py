from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from duo_attn.patch import enable_duo_attention_eval, load_full_attention_heads

from .rekv_core.patch import patch_hf


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
    lm_device = next(model.language_model.parameters()).device
    lm_dtype = next(model.language_model.parameters()).dtype
    vision_device = next(model.vision_tower.parameters()).device
    vision_dtype = next(model.vision_tower.parameters()).dtype

    pixel_values_videos = processor.video_processor(
        video_chunk,
        return_tensors="pt",
    ).pixel_values_videos.to(device=vision_device, dtype=vision_dtype)

    batch_size, frames, channels, height, width = pixel_values_videos.shape
    pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
    vision_feature_layer = model.config.vision_feature_layer
    vision_feature_select_strategy = model.config.vision_feature_select_strategy

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


def maybe_reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def read_peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def greedy_decode_with_cache(
    *,
    language_model,
    tokenizer,
    prompt_text: str,
    past_key_values,
    stop_token_ids: set[int],
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, dict[str, Any]]:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_embeds = language_model.get_input_embeddings()(prompt_ids)
    decode_start = time.perf_counter()
    if max_new_tokens <= 0:
        return "", {
            "ttft_sec": 0.0,
            "answer_latency_sec": 0.0,
            "generated_token_count": 0,
            "prompt_token_count": int(prompt_ids.shape[-1]),
            "prompt_prefill_mode": "inputs_embeds",
        }

    output_ids: list[int] = []
    stop_reason = "max_new_tokens"
    with torch.inference_mode():
        out = language_model(
            inputs_embeds=prompt_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        working_cache = out.past_key_values
        logits = out.logits
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
            logits = out.logits
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
        "prompt_token_count": int(prompt_ids.shape[-1]),
        "prompt_prefill_mode": "inputs_embeds",
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
    def answer_question(self, question: str, metadata: dict[str, Any] | None = None) -> MethodAnswer:
        raise NotImplementedError

    @abstractmethod
    def get_runtime_stats(self) -> dict[str, Any]:
        raise NotImplementedError


class _BaseLlavaStreamingMethod(StreamingMethod):
    def __init__(
        self,
        *,
        pretrained: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.max_new_tokens = int(max_new_tokens)
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
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._encode_init_prompt()

    def ingest_frame(self, frame: np.ndarray, timestamp_sec: float) -> dict[str, Any]:
        video_chunk = np.expand_dims(frame, axis=0)
        start = time.perf_counter()
        with torch.inference_mode():
            video_features = extract_video_features(self.model, self.processor, video_chunk)
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
        prediction, decode_stats = greedy_decode_with_cache(
            language_model=self.model.language_model,
            tokenizer=self.tokenizer,
            prompt_text=prompt_text,
            past_key_values=self._prepare_answer_cache(question, metadata),
            stop_token_ids=self.stop_token_ids,
            max_new_tokens=self.max_new_tokens,
            device=self.device,
        )
        method_stats = self._build_answer_stats(question, metadata)
        method_stats.update(decode_stats)
        method_stats["peak_memory_bytes"] = read_peak_memory_bytes(self.device)
        method_stats["frames_ingested_so_far"] = self.frames_ingested
        return MethodAnswer(prediction=prediction, stats=method_stats)

    @abstractmethod
    def _validate_video_features(self, video_features: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def _prepare_answer_cache(self, question: str, metadata: dict[str, Any]):
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
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
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
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.actual_sparsity = float(attn_config["actual_sparsity"])
        self.learned_threshold = float(attn_config["learned_threshold"])

    def _validate_video_features(self, video_features: torch.Tensor) -> None:
        if video_features.ndim != 3 or video_features.shape[0] != 1:
            raise ValueError(f"Unexpected video feature shape for DuoAttention: {tuple(video_features.shape)}")

    def _prepare_answer_cache(self, question: str, metadata: dict[str, Any]):
        return self.base_cache

    def _build_answer_stats(self, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "attn_dir": self.attn_dir,
            "sink_size": self.sink_size,
            "recent_size": self.recent_size,
            "actual_sparsity": self.actual_sparsity,
            "learned_threshold": self.learned_threshold,
        }


class FullStreamingMethod(_BaseLlavaStreamingMethod):
    method_name = "full_streaming"

    def _validate_video_features(self, video_features: torch.Tensor) -> None:
        if video_features.ndim != 3 or video_features.shape[0] != 1:
            raise ValueError(
                f"Unexpected video feature shape for full streaming: {tuple(video_features.shape)}"
            )

    def _prepare_answer_cache(self, question: str, metadata: dict[str, Any]):
        return self.base_cache

    def _build_answer_stats(self, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"method_name": self.method_name}


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
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
        self.n_local = int(n_local)
        self.retrieve_size = int(retrieve_size)
        self.retrieve_chunk_size = int(retrieve_chunk_size)
        self.n_frame_tokens = int(n_frame_tokens)
        self.fattn = bool(fattn)
        self.pin_memory = bool(pin_memory and self.device.type == "cuda")
        self.model.language_model = patch_hf(
            self.model.language_model,
            n_init=int(self.init_prompt_ids.shape[-1]),
            n_local=self.n_local,
            fattn=self.fattn,
            block_size=self.n_frame_tokens,
            topk=self.retrieve_size,
            chunk_size=self.retrieve_chunk_size,
            max_cached_block=128,
            exc_block_size=self.n_frame_tokens,
            pin_memory=self.pin_memory,
        )
        self.last_retrieval_stats: dict[str, Any] = {}

    def _validate_video_features(self, video_features: torch.Tensor) -> None:
        if video_features.ndim != 3 or video_features.shape[0] != 1:
            raise ValueError(f"Unexpected video feature shape for ReKV: {tuple(video_features.shape)}")
        if video_features.shape[1] != self.n_frame_tokens:
            raise ValueError(
                "ReKV expects one ingested frame to map to exactly "
                f"{self.n_frame_tokens} visual tokens, got {video_features.shape[1]}."
            )

    def _prepare_answer_cache(self, question: str, metadata: dict[str, Any]):
        question_ids = self.tokenizer(question, return_tensors="pt").input_ids.to(self.device)
        retrieval_start = time.perf_counter()
        for layer_kv in self.base_cache:
            layer_kv.set_retrieval()

        with torch.inference_mode():
            out = self.model.language_model(
                input_ids=question_ids,
                use_cache=True,
                past_key_values=self.base_cache,
            )
        retrieval_past = out.past_key_values
        retrieval_latency_sec = time.perf_counter() - retrieval_start

        per_layer_retrieved_counts: list[int] = []
        per_layer_global_blocks: list[int] = []
        for layer_kv in self.base_cache:
            retrieved_indices = layer_kv.retrieved_block_indices or []
            if retrieved_indices and retrieved_indices[0] is not None:
                per_layer_retrieved_counts.append(len(retrieved_indices[0]))
            else:
                per_layer_retrieved_counts.append(0)
            per_layer_global_blocks.append(int(getattr(layer_kv, "num_global_block", 0)))
            layer_kv.reset_retrieval()

        self.last_retrieval_stats = {
            "retrieval_latency_sec": retrieval_latency_sec,
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
        }
        return retrieval_past

    def _build_answer_stats(self, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "n_local": self.n_local,
            "retrieve_size": self.retrieve_size,
            "retrieve_chunk_size": self.retrieve_chunk_size,
            "n_frame_tokens": self.n_frame_tokens,
            "fattn": self.fattn,
            **self.last_retrieval_stats,
        }


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
    ) -> None:
        _BaseLlavaStreamingMethod.__init__(
            self,
            pretrained=pretrained,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
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
        for layer in self.model.language_model.model.layers:
            layer.self_attn.rekv_duo_enabled = True

        self.attn_dir = attn_config["attn_dir"]
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.actual_sparsity = float(attn_config["actual_sparsity"])
        self.learned_threshold = float(attn_config["learned_threshold"])

        self.n_local = int(n_local)
        self.retrieve_size = int(retrieve_size)
        self.retrieve_chunk_size = int(retrieve_chunk_size)
        self.n_frame_tokens = int(n_frame_tokens)
        self.fattn = bool(fattn)
        self.pin_memory = bool(pin_memory and self.device.type == "cuda")
        self.model.language_model = patch_hf(
            self.model.language_model,
            n_init=int(self.init_prompt_ids.shape[-1]),
            n_local=self.n_local,
            fattn=self.fattn,
            block_size=self.n_frame_tokens,
            topk=self.retrieve_size,
            chunk_size=self.retrieve_chunk_size,
            max_cached_block=128,
            exc_block_size=self.n_frame_tokens,
            pin_memory=self.pin_memory,
        )
        self.last_retrieval_stats: dict[str, Any] = {}

    def _build_answer_stats(self, question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "integration_mode": "rekv_memory_plus_duo_lm",
            "retrieval_policy": "rekv_native",
            "answer_attention_policy": "duo_post_retrieval",
            "attn_dir": self.attn_dir,
            "sink_size": self.sink_size,
            "recent_size": self.recent_size,
            "actual_sparsity": self.actual_sparsity,
            "learned_threshold": self.learned_threshold,
            "n_local": self.n_local,
            "retrieve_size": self.retrieve_size,
            "retrieve_chunk_size": self.retrieve_chunk_size,
            "n_frame_tokens": self.n_frame_tokens,
            "fattn": self.fattn,
            **self.last_retrieval_stats,
        }


def build_method_from_args(args) -> StreamingMethod:
    shared_kwargs = {
        "pretrained": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
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
