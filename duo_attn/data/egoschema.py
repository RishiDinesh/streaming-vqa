import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseVideoQADataset


class EgoSchemaDataset(BaseVideoQADataset):
    # Video QA dataset for EgoSchema-style question files.

    def __init__(
        self,
        video_root: str,
        annotation_path: str,
        processor: Optional[Any] = None,
        model_id: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        num_frames: int = 8,
        max_length: int = 2048,
        use_chat_template: bool = True,
        include_options_in_question: bool = False,
        default_video_ext: str = ".mp4",
    ):
        if not annotation_path:
            raise ValueError("`annotation_path` is required.")

        self.annotation_path = str(annotation_path)
        self.include_options_in_question = bool(include_options_in_question)
        self.default_video_ext = str(default_video_ext)

        super().__init__(
            video_root=video_root,
            processor=processor,
            model_id=model_id,
            num_frames=num_frames,
            max_length=max_length,
            use_chat_template=use_chat_template,
        )

        self.samples = self._load_annotations(self.annotation_path)
        self._dataset_length = len(self.samples)

    def _build_sample(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        question = str(sample["question"]).strip()
        options = sample.get("options", [])

        if self.include_options_in_question and options:
            option_lines = [f"({i}) {opt}" for i, opt in enumerate(options)]
            option_block = "\n".join(option_lines)
            question = (
                f"{question}\nChoices:\n"
                f"{option_block}\n"
                "Respond with the best option text."
            )

        target = f"\nAnswer: {str(sample['gt']).strip()}"

        video_path = self._resolve_video_path(sample["video"])
        frames = self._decode_and_sample_frames(video_path)

        prefix_text = self._build_prefix_text(question)
        full_text = f"{prefix_text}{target}"

        extras = {
            "q_uid": sample.get("q_uid", ""),
            "video": sample["video"],
        }

        return {
            "frames": frames,
            "prefix_text": prefix_text,
            "full_text": full_text,
            "extras": extras,
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
                raw_samples = None
                for key in ("data", "samples", "annotations", "questions"):
                    maybe_list = payload.get(key)
                    if isinstance(maybe_list, list):
                        raw_samples = maybe_list
                        break
                if raw_samples is None:
                    raw_samples = [payload]
            else:
                raise ValueError(
                    "Unsupported annotation payload. Expected JSON list/dict or JSONL."
                )

        valid_samples: List[Dict[str, Any]] = []
        for sample in raw_samples:
            if not isinstance(sample, dict):
                continue

            question = self._extract_question(sample)
            if not question:
                continue

            video = self._extract_video(sample)
            if not video:
                continue

            options = self._extract_options(sample)
            answer_idx = self._extract_answer_index(sample)
            gt = self._extract_answer_text(sample, options, answer_idx)

            valid_samples.append(
                {
                    "q_uid": sample.get("q_uid", ""),
                    "video": video,
                    "question": question,
                    "options": options,
                    "gt": gt,
                }
            )

        if not valid_samples:
            raise ValueError(
                "No valid EgoSchema samples found. Expected question + video fields."
            )

        return valid_samples

    def _extract_question(self, sample: Dict[str, Any]) -> str:
        for key in ("question", "query", "prompt"):
            value = sample.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _extract_video(self, sample: Dict[str, Any]) -> str:
        value = None
        for key in ("video", "video_path", "video_name", "video_id", "q_uid"):
            candidate = sample.get(key)
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                value = text
                break

        if value is None:
            return ""

        if "." in value.split("/")[-1]:
            return value
        return f"{value}{self.default_video_ext}"

    def _extract_options(self, sample: Dict[str, Any]) -> List[str]:
        for key in ("options", "choices", "candidates"):
            value = sample.get(key)
            if isinstance(value, list):
                options = [str(v).strip() for v in value if str(v).strip()]
                if options:
                    return options

        numbered: List[tuple[int, str]] = []
        for key, value in sample.items():
            low = str(key).strip().lower()
            if not low.startswith("option"):
                continue
            suffix = low.replace("option", "").strip(" _-")
            if not suffix.isdigit():
                continue
            text = str(value).strip()
            if not text:
                continue
            numbered.append((int(suffix), text))

        if numbered:
            numbered.sort(key=lambda x: x[0])
            return [text for _, text in numbered]

        return []

    def _extract_answer_index(self, sample: Dict[str, Any]) -> Optional[int]:
        for key in ("answer", "label", "answer_idx", "correct_idx", "target"):
            value = sample.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        return None

    def _extract_answer_text(
        self,
        sample: Dict[str, Any],
        options: List[str],
        answer_idx: Optional[int],
    ) -> str:
        gt = sample.get("gt")
        if gt is not None and str(gt).strip():
            return str(gt).strip()

        if answer_idx is not None and 0 <= answer_idx < len(options):
            return options[answer_idx]

        for key in ("answer_text", "answer_str"):
            value = sample.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

        return "N/A"
