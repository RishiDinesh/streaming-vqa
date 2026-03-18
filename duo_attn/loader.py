import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VideoQADataset(Dataset):
    """
    Video QA dataset formatted for LLaVA-OneVision-style training.

    Each sample returns:
    - `input_ids`: tokenized multimodal prompt + answer
    - `labels`: `-100` over prompt tokens and target ids over answer tokens
    - processor-produced vision tensors (for example `pixel_values_videos`)
    """

    def __init__(
        self,
        video_root: str,
        annotation_path: str,
        processor: Optional[Any] = None,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        num_frames: int = 8,
        max_length: int = 2048,
        use_chat_template: bool = True,
        answer_prefix: str = "The secret word is: ",
    ):
        super().__init__()

        if not video_root:
            raise ValueError("`video_root` is required.")
        if not annotation_path:
            raise ValueError("`annotation_path` is required.")
        if num_frames <= 0:
            raise ValueError("`num_frames` must be > 0.")
        if max_length <= 0:
            raise ValueError("`max_length` must be > 0.")

        self.video_root = os.path.abspath(str(video_root))
        self.annotation_path = str(annotation_path)
        self.num_frames = int(num_frames)
        self.max_length = int(max_length)
        self.use_chat_template = bool(use_chat_template)
        self.answer_prefix = answer_prefix

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
        self.samples = self._load_annotations(self.annotation_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        question = str(sample["question"]).strip()
        target = str(sample["gt"]).strip()

        video_path = self._resolve_video_path(sample["video"])
        frames = self._decode_and_sample_frames(video_path)

        prefix_text = self._build_prefix_text(question)
        full_text = f"{prefix_text}{target}"

        return self._build_model_inputs(frames, prefix_text, full_text)

    def _load_annotations(self, annotation_path: str) -> List[Dict[str, Any]]:
        path = Path(annotation_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                raw_samples = [json.loads(line) for line in f if line.strip()]
        else:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            if isinstance(payload, list):
                raw_samples = payload
            elif isinstance(payload, dict):
                if all(k in payload for k in ("video", "question", "gt")):
                    raw_samples = [payload]
                else:
                    raw_samples = None
                    for key in ("data", "samples", "annotations"):
                        maybe_list = payload.get(key)
                        if isinstance(maybe_list, list):
                            raw_samples = maybe_list
                            break
                    if raw_samples is None:
                        raise ValueError(
                            "Unsupported JSON format. Expected a list of samples or one of "
                            "the wrapped keys: data/samples/annotations."
                        )
            else:
                raise ValueError(
                    "Unsupported annotation payload. Expected JSON list/dict or JSONL."
                )

        valid_samples: List[Dict[str, Any]] = []
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue
            if "video" not in sample or "question" not in sample or "gt" not in sample:
                continue
            valid_samples.append(sample)

        if not valid_samples:
            raise ValueError(
                "No valid samples found in annotation file. Each sample must include "
                "`video`, `question`, and `gt`."
            )

        return valid_samples

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
                return f"{prompt}{self.answer_prefix}"
            except Exception:
                pass

        return f"{self.video_token}\n{question}\n{self.answer_prefix}"

    def _build_model_inputs(
        self, frames: Sequence[Any], prefix_text: str, full_text: str
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
            print("Warning: multimodal encoding failed for sample. Falling back to text-only encoding.")
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
        self, prefix_text: str, full_text: str
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


@dataclass
class VideoQACollator:
    tokenizer: Any

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must provide pad_token_id or eos_token_id.")

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        ret = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
        }

        for key in instances[0].keys():
            if key in ret:
                continue
            values = [instance[key] for instance in instances]
            if not all(torch.is_tensor(v) for v in values):
                continue

            if all(v.shape == values[0].shape for v in values):
                ret[key] = torch.stack(values)
                continue

            if values[0].ndim == 1:
                ret[key] = torch.nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=0
                )
                continue

            raise ValueError(
                f"Cannot collate key '{key}' with shapes: "
                f"{[tuple(v.shape) for v in values]}"
            )

        return ret


def create_video_qa_dataloader(
    video_root: str,
    annotation_path: str,
    processor: Optional[Any] = None,
    model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    num_frames: int = 8,
    max_length: int = 2048,
    use_chat_template: bool = True,
    answer_prefix: str = "The secret word is: ",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    dataset = VideoQADataset(
        video_root=video_root,
        annotation_path=annotation_path,
        processor=processor,
        model_id=model_id,
        num_frames=num_frames,
        max_length=max_length,
        use_chat_template=use_chat_template,
        answer_prefix=answer_prefix,
    )
    collator = VideoQACollator(tokenizer=dataset.tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )
