from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from .common import StreamingConversation, StreamingVideoSample


DEFAULT_RVS_REPO_ID = "Becomebright/RVS"
DEFAULT_RVS_SUBSET = "ego"
DEFAULT_RVS_ANNOTATION = "ego/ego4d_oe.json"


@dataclass(frozen=True)
class SampledVideo:
    video_path: str
    native_fps: float
    sampled_frame_indices: list[int]
    sampled_timestamps_sec: list[float]
    _reader: Any

    def get_frame(self, sampled_index: int) -> np.ndarray:
        frame_index = self.sampled_frame_indices[sampled_index]
        frame_batch = self._reader.get_batch([frame_index]).asnumpy()
        return frame_batch[0]


class RVSEgoDataset:
    def __init__(
        self,
        *,
        annotation_path: str | None = None,
        video_root: str | None = None,
        hf_repo_id: str = DEFAULT_RVS_REPO_ID,
        hf_subset: str = DEFAULT_RVS_SUBSET,
        allow_hf_video_download: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        self.annotation_path = annotation_path
        self.video_root = os.path.abspath(video_root) if video_root else None
        self.hf_repo_id = hf_repo_id
        self.hf_subset = hf_subset
        self.allow_hf_video_download = bool(allow_hf_video_download)
        self.cache_dir = cache_dir
        self._annotation_file = self._resolve_annotation_file()

    def _resolve_annotation_file(self) -> str:
        if self.annotation_path:
            path = os.path.abspath(self.annotation_path)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Annotation file not found: {path}")
            return path

        return hf_hub_download(
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            filename=DEFAULT_RVS_ANNOTATION,
            cache_dir=self.cache_dir,
        )

    def _normalize_rel_path(self, path_value: Any) -> str:
        normalized = str(path_value).strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        while normalized.startswith("/"):
            normalized = normalized[1:]
        return normalized

    def _hf_video_candidates(self, raw_path: str, normalized: str) -> list[str]:
        candidates = [normalized]
        if normalized.startswith("data/rvs/"):
            candidates.append(normalized[len("data/rvs/"):])
        if normalized.startswith("data/"):
            candidates.append(normalized[len("data/"):])
        if normalized.startswith(f"{self.hf_subset}/"):
            candidates.append(normalized)
        candidates.append(os.path.join(self.hf_subset, "videos", os.path.basename(normalized)))
        candidates.append(os.path.basename(normalized))

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            clean = candidate.strip().replace("\\", "/")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)
        return deduped

    def _resolve_video_path(self, video_value: Any) -> str:
        raw_path = str(video_value).strip()
        normalized = self._normalize_rel_path(raw_path)
        candidates: list[str] = []

        def maybe_add(path: str | None) -> None:
            if path:
                candidates.append(os.path.abspath(path))

        maybe_add(raw_path if os.path.isabs(raw_path) else None)
        maybe_add(os.path.join(os.path.dirname(self._annotation_file), raw_path))
        maybe_add(os.path.join(os.path.dirname(self._annotation_file), normalized))
        if self.video_root:
            maybe_add(os.path.join(self.video_root, raw_path))
            maybe_add(os.path.join(self.video_root, normalized))
            maybe_add(os.path.join(self.video_root, os.path.basename(normalized)))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        if self.allow_hf_video_download:
            for filename in self._hf_video_candidates(raw_path, normalized):
                try:
                    path = hf_hub_download(
                        repo_id=self.hf_repo_id,
                        repo_type="dataset",
                        filename=filename,
                        cache_dir=self.cache_dir,
                    )
                except Exception:
                    continue
                if os.path.exists(path):
                    return os.path.abspath(path)

        raise FileNotFoundError(
            "Could not resolve video file for RVS-Ego sample.\n"
            f"annotation_file: {self._annotation_file}\n"
            f"video_root: {self.video_root}\n"
            f"original_value: {video_value}"
        )

    def load(
        self,
        *,
        video_id: str | None = None,
        video_index: int | None = None,
        max_videos: int | None = None,
    ) -> list[StreamingVideoSample]:
        with open(self._annotation_file, "r", encoding="utf-8") as handle:
            raw_records = json.load(handle)

        if video_id is not None:
            raw_records = [record for record in raw_records if str(record["video_id"]) == video_id]
        if video_index is not None:
            raw_records = raw_records[video_index : video_index + 1]
        if max_videos is not None:
            raw_records = raw_records[:max_videos]

        samples: list[StreamingVideoSample] = []
        for record_idx, record in enumerate(raw_records):
            conversations = sorted(
                record.get("conversations", []),
                key=lambda item: float(item["end_time"]),
            )
            normalized_conversations = [
                StreamingConversation(
                    question=str(item["question"]).strip(),
                    answer=str(item["answer"]).strip(),
                    start_time=float(item["start_time"]),
                    end_time=float(item["end_time"]),
                    extra_metadata={
                        key: value
                        for key, value in item.items()
                        if key not in {"question", "answer", "start_time", "end_time"}
                    },
                )
                for item in conversations
            ]
            video_path = self._resolve_video_path(record["video_path"])
            video_id = str(record["video_id"])
            samples.append(
                StreamingVideoSample(
                    sample_id=f"{video_id}-{record_idx}",
                    video_id=video_id,
                    video_path=video_path,
                    duration=float(record["duration"]),
                    conversations=normalized_conversations,
                    extra_metadata={
                        key: value
                        for key, value in record.items()
                        if key
                        not in {"video_id", "video_path", "duration", "conversations"}
                    },
                )
            )

        return samples


def sample_video_frames(video_path: str, sample_fps: float) -> SampledVideo:
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be > 0, got {sample_fps}")

    from decord import VideoReader, cpu

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    native_fps = float(reader.get_avg_fps())
    if native_fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {video_path}")

    stride = max(int(native_fps / sample_fps), 1)
    frame_indices = list(range(0, len(reader), stride))
    timestamps_sec = [frame_index / native_fps for frame_index in frame_indices]

    return SampledVideo(
        video_path=os.path.abspath(video_path),
        native_fps=native_fps,
        sampled_frame_indices=frame_indices,
        sampled_timestamps_sec=timestamps_sec,
        _reader=reader,
    )
