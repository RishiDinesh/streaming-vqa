import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseVideoQADataset(Dataset, ABC):
    """
    Shared video QA dataset utilities for LLaVA-OneVision-style training.

    Subclasses only need to provide `_build_sample(index)` and set
    `self._dataset_length` during initialization.
    """

    def __init__(
        self,
        video_root: str,
        processor: Optional[Any] = None,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        num_frames: int = 8,
        max_length: int = 2048,
        use_chat_template: bool = True
    ):
        super().__init__()

        if not video_root:
            raise ValueError("`video_root` is required.")
        if num_frames <= 0:
            raise ValueError("`num_frames` must be > 0.")
        if max_length <= 0:
            raise ValueError("`max_length` must be > 0.")

        self.video_root = os.path.abspath(str(video_root))
        self.num_frames = int(num_frames)
        self.max_length = int(max_length)
        self.use_chat_template = bool(use_chat_template)

        if processor is None:
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "transformers is required when `processor` is not provided."
                ) from exc
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self.processor = processor
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("`processor.tokenizer` is required.")

        self.video_token = self._infer_video_token()
        self._dataset_length: Optional[int] = None

    def __len__(self) -> int:
        if self._dataset_length is None:
            raise NotImplementedError(
                "Subclasses must set `self._dataset_length` during initialization."
            )
        return int(self._dataset_length)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self._build_sample(index)
        frames = sample["frames"]
        prefix_text = sample["prefix_text"]
        full_text = sample["full_text"]

        ret = self._build_model_inputs(frames, prefix_text, full_text)
        extras = sample.get("extras", None)
        if isinstance(extras, dict):
            ret.update(extras)
        return ret

    @abstractmethod
    def _build_sample(self, index: int) -> Dict[str, Any]:
        """Return a dict with keys: frames, prefix_text, full_text, optional extras."""

    def _infer_video_token(self) -> str:
        for attr in ("video_token", "vision_token", "image_token"):
            token = getattr(self.tokenizer, attr, None)
            if token:
                return token
            token = getattr(self.processor, attr, None)
            if token:
                return token

        extra_tokens = getattr(self.tokenizer, "additional_special_tokens", None)
        if extra_tokens:
            for token in extra_tokens:
                low = token.lower()
                if "video" in low or "vision" in low:
                    return token
            for token in extra_tokens:
                if "image" in token.lower():
                    return token

        return "<video>"

    def _normalize_rel_path(self, path_value: Any) -> str:
        normalized = str(path_value).strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        while normalized.startswith("/"):
            normalized = normalized[1:]
        return normalized

    def _resolve_video_path(self, video_value: Any) -> str:
        raw_path = str(video_value).strip()
        tried: List[str] = []

        def _maybe_return(path: str) -> Optional[str]:
            if not path:
                return None
            tried.append(path)
            if os.path.exists(path):
                return os.path.abspath(path)
            return None

        resolved = _maybe_return(raw_path)
        if resolved is not None:
            return resolved

        normalized = self._normalize_rel_path(raw_path)
        candidates = [
            os.path.join(self.video_root, raw_path),
            os.path.join(self.video_root, normalized),
            os.path.join(self.video_root, os.path.basename(normalized)),
        ]

        for candidate in candidates:
            resolved = _maybe_return(candidate)
            if resolved is not None:
                return resolved

        raise FileNotFoundError(
            "Could not resolve video file.\n"
            f"video_root: {self.video_root}\n"
            f"original value: {video_value}\n"
            f"attempted: {tried}"
        )

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            raise ValueError("Video has zero frames.")
        if self.num_frames == 1:
            return [0]
        if total_frames == 1:
            return [0] * self.num_frames

        return (
            torch.linspace(0, total_frames - 1, steps=self.num_frames)
            .round()
            .long()
            .tolist()
        )

    def _decode_and_sample_frames(self, video_path: str):
        errors = []
        for decoder in (
            self._decode_with_decord,
            self._decode_with_torchvision,
            self._decode_with_opencv,
        ):
            try:
                return decoder(video_path)
            except Exception as exc:
                errors.append(f"{decoder.__name__}: {exc}")

        raise RuntimeError(
            f"Unable to decode video '{video_path}'. Tried decord/torchvision/opencv. "
            + "; ".join(errors)
        )

    def _decode_with_decord(self, video_path: str):
        from decord import VideoReader, cpu

        reader = VideoReader(video_path, ctx=cpu(0))
        frame_indices = self._sample_frame_indices(len(reader))
        frames = reader.get_batch(frame_indices).asnumpy()
        return self._to_pil_frames(frames)

    def _decode_with_torchvision(self, video_path: str):
        from torchvision.io import read_video

        video, _, _ = read_video(video_path, pts_unit="sec", output_format="TCHW")
        if video.ndim != 4 or video.shape[0] == 0:
            raise ValueError("torchvision read_video returned no frames.")

        frame_indices = torch.tensor(
            self._sample_frame_indices(video.shape[0]), dtype=torch.long
        )
        sampled = video.index_select(0, frame_indices)
        frames = sampled.permute(0, 2, 3, 1).cpu().numpy()
        return self._to_pil_frames(frames)

    def _decode_with_opencv(self, video_path: str):
        import cv2

        capture = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()

        if not frames:
            raise ValueError("OpenCV decoder returned no frames.")

        frame_indices = self._sample_frame_indices(len(frames))
        sampled = [frames[idx] for idx in frame_indices]
        return self._to_pil_frames(np.stack(sampled))

    def _to_pil_frames(self, frames):
        from PIL import Image

        if isinstance(frames, list) and frames and hasattr(frames[0], "size"):
            return [frame.convert("RGB") for frame in frames]

        if torch.is_tensor(frames):
            frames = frames.detach().cpu().numpy()
        if isinstance(frames, list):
            frames = np.stack(frames)

        if not isinstance(frames, np.ndarray) or frames.ndim != 4:
            raise ValueError(
                "Expected decoded frames in (T,H,W,C), got "
                f"{type(frames)} shape={getattr(frames, 'shape', None)}"
            )

        pil_frames = []
        for frame in frames:
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame_uint8).convert("RGB"))
        return pil_frames

    def _build_prefix_text(self, question: str) -> str:
        if self.use_chat_template and hasattr(self.processor, "apply_chat_template"):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            try:
                prompt = self.processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception:
                pass

        return f"{self.video_token}\n{question}"

    def _build_model_inputs(
        self,
        frames: Sequence[Any],
        prefix_text: str,
        full_text: str,
    ) -> Dict[str, torch.Tensor]:
        try:
            full_outputs = self._encode_multimodal(frames, full_text)
            prefix_outputs = self._encode_multimodal(frames, prefix_text)
            input_ids = self._extract_input_ids(full_outputs)
            prefix_ids = self._extract_input_ids(prefix_outputs)

            labels = input_ids.clone()
            labels[: self._common_prefix_length(input_ids, prefix_ids)] = -100

            ret = self._collect_tensor_outputs(full_outputs)
            ret["input_ids"] = input_ids
            ret["labels"] = labels
            return ret
        except RuntimeError:
            print(
                "Warning: multimodal encoding failed for sample. "
                "Falling back to text-only encoding."
            )
            input_ids, labels = self._encode_text_fallback(prefix_text, full_text)
            vision_tensors = self._encode_vision_only(frames)
            ret = {"input_ids": input_ids, "labels": labels}
            ret.update(vision_tensors)
            return ret

    def _encode_multimodal(self, frames: Sequence[Any], text: str) -> Dict[str, Any]:
        attempts = [
            {"text": text, "videos": [frames]},
            {"text": [text], "videos": [frames]},
            {"text": text, "videos": frames},
            {"text": [text], "videos": frames},
        ]

        last_error = None
        for kwargs in attempts:
            for with_length_controls in (True, False):
                try:
                    processor_kwargs = {"return_tensors": "pt", **kwargs}
                    if with_length_controls:
                        processor_kwargs["truncation"] = True
                        processor_kwargs["max_length"] = self.max_length
                    outputs = self.processor(**processor_kwargs)
                    if outputs.get("input_ids", None) is not None:
                        return outputs
                except Exception as exc:
                    last_error = exc

        raise RuntimeError(
            "Unable to jointly encode text and video with the processor."
        ) from last_error

    def _encode_text_fallback(
        self,
        prefix_text: str,
        full_text: str,
    ) -> Sequence[torch.Tensor]:
        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prefix_encoding = self.tokenizer(
            prefix_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = full_encoding["input_ids"][0].long()
        prefix_ids = prefix_encoding["input_ids"][0].long()
        labels = input_ids.clone()
        labels[: self._common_prefix_length(input_ids, prefix_ids)] = -100
        return input_ids, labels

    def _encode_vision_only(self, frames: Sequence[Any]) -> Dict[str, torch.Tensor]:
        attempts = [
            {"videos": [frames]},
            {"videos": frames},
        ]

        last_error = None
        for kwargs in attempts:
            try:
                outputs = self.processor(return_tensors="pt", **kwargs)
                tensors = self._collect_tensor_outputs(outputs)
                if tensors:
                    return tensors
            except Exception as exc:
                last_error = exc

        raise RuntimeError("Unable to encode video frames with processor.") from last_error

    def _extract_input_ids(self, outputs: Dict[str, Any]) -> torch.Tensor:
        input_ids = outputs.get("input_ids", None)
        if input_ids is None:
            raise RuntimeError("Processor outputs did not include input_ids.")

        tensor = self._to_tensor(input_ids)
        if tensor is None:
            raise RuntimeError("Could not convert input_ids to tensor.")

        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1:
            raise RuntimeError(f"Expected 1D input_ids, got {tuple(tensor.shape)}.")
        return tensor.long()

    def _collect_tensor_outputs(self, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            if key in {"input_ids", "labels", "attention_mask", "token_type_ids"}:
                continue

            tensor = self._to_tensor(value)
            if tensor is None:
                continue
            if tensor.ndim > 0 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            ret[key] = tensor
        return ret

    def _to_tensor(self, value: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(value):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, (int, float, bool)):
            return torch.tensor(value)
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            if all(torch.is_tensor(v) for v in value):
                return torch.stack(list(value))
            if all(isinstance(v, np.ndarray) for v in value):
                return torch.from_numpy(np.stack(value))
            try:
                return torch.tensor(value)
            except Exception:
                return None
        return None

    def _common_prefix_length(self, left: torch.Tensor, right: torch.Tensor) -> int:
        max_len = min(left.shape[0], right.shape[0])
        if max_len == 0:
            return 0

        eq = left[:max_len].eq(right[:max_len])
        mismatch = torch.nonzero(~eq, as_tuple=False)
        if mismatch.numel() == 0:
            return max_len
        return int(mismatch[0].item())
