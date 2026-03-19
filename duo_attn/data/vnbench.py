import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseVideoQADataset


class VideoQADataset(BaseVideoQADataset):
    """Video QA dataset backed by annotation files (VNBench-style format)."""

    def __init__(
        self,
        video_root: str,
        annotation_path: str,
        processor: Optional[Any] = None,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        num_frames: int = 8,
        max_length: int = 2048,
        use_chat_template: bool = True
    ):
        if not annotation_path:
            raise ValueError("`annotation_path` is required.")

        self.annotation_path = str(annotation_path)

        super().__init__(
            video_root=video_root,
            processor=processor,
            model_id=model_id,
            num_frames=num_frames,
            max_length=max_length,
            use_chat_template=use_chat_template
        )

        self.samples = self._load_annotations(self.annotation_path)
        self._dataset_length = len(self.samples)

    def _build_sample(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        question = str(sample["question"]).strip()
        target = f"\nThe secret word is: {str(sample['gt']).strip()}"

        video_path = self._resolve_video_path(sample["video"])
        frames = self._decode_and_sample_frames(video_path)

        prefix_text = self._build_prefix_text(question)
        full_text = f"{prefix_text}{target}"

        return {
            "frames": frames,
            "prefix_text": prefix_text,
            "full_text": full_text,
        }

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
