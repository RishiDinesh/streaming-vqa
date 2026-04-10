from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from .common import StreamingConversation, StreamingVideoSample


DEFAULT_RVS_REPO_ID = "Becomebright/RVS"

RVS_DATASET_CONFIGS = {
    "rvs_ego": {
        "subset": "ego",
        "annotation": "ego/ego4d_oe.json",
        "label": "RVS-Ego",
    },
    "rvs_movie": {
        "subset": "movie",
        "annotation": "movie/movienet_oe.json",
        "label": "RVS-Movie",
    },
}


@dataclass(frozen=True)
class SampledVideo:
    video_path: str
    native_fps: float
    sampling_base_fps: int
    num_source_frames: int
    sampled_frame_indices: list[int]
    sampled_timestamps_sec: list[float]
    _reader: Any

    def get_frame(self, sampled_index: int) -> np.ndarray:
        frame_index = self.sampled_frame_indices[sampled_index]
        if hasattr(self._reader, "get_batch"):
            try:
                frame_batch = self._reader.get_batch([frame_index]).asnumpy()
                return frame_batch[0]
            except Exception:
                try:
                    from decord import VideoReader, cpu
                    reader_st = VideoReader(self.video_path, ctx=cpu(0), num_threads=1)
                    return reader_st.get_batch([frame_index]).asnumpy()[0]
                except Exception:
                    pass
        if hasattr(self._reader, "get_data"):
            return np.asarray(self._reader.get_data(frame_index))
        try:
            import imageio.v2 as imageio
            reader_io = imageio.get_reader(self.video_path)
            frame = np.asarray(reader_io.get_data(frame_index))
            reader_io.close()
            return frame
        except Exception:
            pass
        return np.asarray(self._reader[frame_index])

    def get_frames(self, sampled_indices: list[int]) -> np.ndarray:
        if not sampled_indices:
            return np.empty((0,), dtype=np.uint8)
        frame_indices = [self.sampled_frame_indices[index] for index in sampled_indices]
        if hasattr(self._reader, "get_batch"):
            try:
                return self._reader.get_batch(frame_indices).asnumpy()
            except Exception:
                # Some decord/FFmpeg builds fail on batched reads for specific mp4s
                # (threaded decoder crash). Try one-at-a-time with a fresh
                # single-threaded reader, then fall back to imageio.
                try:
                    from decord import VideoReader, cpu
                    reader_st = VideoReader(self.video_path, ctx=cpu(0), num_threads=1)
                    return np.stack(
                        [reader_st.get_batch([fi]).asnumpy()[0] for fi in frame_indices], axis=0
                    )
                except Exception:
                    pass
        if hasattr(self._reader, "get_data"):
            return np.stack([np.asarray(self._reader.get_data(frame_index)) for frame_index in frame_indices], axis=0)
        # Last resort: imageio sequential decode
        try:
            import imageio.v2 as imageio
            reader_io = imageio.get_reader(self.video_path)
            frames = [np.asarray(reader_io.get_data(fi)) for fi in frame_indices]
            reader_io.close()
            return np.stack(frames, axis=0)
        except Exception:
            pass
        return np.stack([np.asarray(self._reader[frame_index]) for frame_index in frame_indices], axis=0)


class RVSDataset:
    def __init__(
        self,
        *,
        dataset_name: str,
        annotation_path: str | None = None,
        video_root: str | None = None,
        hf_repo_id: str = DEFAULT_RVS_REPO_ID,
        allow_hf_video_download: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        if dataset_name not in RVS_DATASET_CONFIGS:
            raise ValueError(
                f"Unsupported dataset_name={dataset_name!r}; expected one of {sorted(RVS_DATASET_CONFIGS)}"
            )
        config = RVS_DATASET_CONFIGS[dataset_name]
        self.dataset_name = dataset_name
        self.dataset_label = str(config["label"])
        self.annotation_path = annotation_path
        self.video_root = os.path.abspath(video_root) if video_root else None
        self.hf_repo_id = hf_repo_id
        self.hf_subset = str(config["subset"])
        self.annotation_filename = str(config["annotation"])
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
            filename=self.annotation_filename,
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
            f"Could not resolve video file for {self.dataset_label} sample.\n"
            f"annotation_file: {self._annotation_file}\n"
            f"video_root: {self.video_root}\n"
            f"original_value: {video_value}"
        )

    def load(
        self,
        *,
        video_id: str | None = None,
        video_index: int | None = None,
        video_offset: int = 0,
        max_videos: int | None = None,
    ) -> list[StreamingVideoSample]:
        with open(self._annotation_file, "r", encoding="utf-8") as handle:
            raw_records = json.load(handle)

        if video_id is not None:
            raw_records = [record for record in raw_records if str(record["video_id"]) == video_id]
        if video_index is not None:
            raw_records = raw_records[video_index : video_index + 1]
        elif video_offset > 0:
            raw_records = raw_records[video_offset:]
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
            sample_video_id = str(record["video_id"])
            video_path = self._resolve_video_path(record["video_path"])
            samples.append(
                StreamingVideoSample(
                    sample_id=f"{sample_video_id}-{record_idx}",
                    video_id=sample_video_id,
                    video_path=video_path,
                    duration=float(record["duration"]),
                    conversations=normalized_conversations,
                    extra_metadata={
                        key: value
                        for key, value in record.items()
                        if key not in {"video_id", "video_path", "duration", "conversations"}
                    },
                )
            )

        return samples


class RVSEgoDataset(RVSDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(dataset_name="rvs_ego", **kwargs)


class RVSMovieDataset(RVSDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(dataset_name="rvs_movie", **kwargs)


def build_dataset_from_args(args) -> RVSDataset:
    dataset_kwargs = {
        "annotation_path": args.annotation_path,
        "video_root": args.video_root,
        "hf_repo_id": args.hf_repo_id,
        "allow_hf_video_download": args.allow_hf_video_download,
    }
    if args.dataset == "rvs_ego":
        return RVSEgoDataset(**dataset_kwargs)
    if args.dataset == "rvs_movie":
        return RVSMovieDataset(**dataset_kwargs)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def _sample_numpy_video(
    video_path: str,
    sample_fps: float,
    duration_sec: float | None,
) -> SampledVideo:
    array = np.load(video_path, mmap_mode="r")
    if array.ndim != 4:
        raise ValueError(f"Expected 4D numpy video array at {video_path}, got shape {array.shape}")
    if duration_sec is None or duration_sec <= 0:
        raise ValueError(
            "duration_sec must be provided for numpy-backed videos so sampling stays causal."
        )

    native_fps = float(array.shape[0] / duration_sec)
    sampling_base_fps = max(int(round(native_fps)), 1)
    stride = max(int(sampling_base_fps / sample_fps), 1)
    frame_indices = list(range(0, array.shape[0], stride))
    timestamps_sec = [frame_index / sampling_base_fps for frame_index in frame_indices]

    return SampledVideo(
        video_path=os.path.abspath(video_path),
        native_fps=native_fps,
        sampling_base_fps=sampling_base_fps,
        num_source_frames=int(array.shape[0]),
        sampled_frame_indices=frame_indices,
        sampled_timestamps_sec=timestamps_sec,
        _reader=array,
    )


def _sample_decord_video(
    video_path: str,
    sample_fps: float,
    *,
    decode_threads: int = 1,
) -> SampledVideo:
    try:
        from decord import VideoReader, cpu
    except ModuleNotFoundError:
        return _sample_imageio_video(video_path, sample_fps)

    reader = VideoReader(video_path, ctx=cpu(0), num_threads=max(int(decode_threads), 1))
    native_fps = float(reader.get_avg_fps())
    if native_fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {video_path}")

    sampling_base_fps = max(int(round(native_fps)), 1)
    stride = max(int(sampling_base_fps / sample_fps), 1)
    frame_indices = list(range(0, len(reader), stride))
    timestamps_sec = [frame_index / sampling_base_fps for frame_index in frame_indices]

    return SampledVideo(
        video_path=os.path.abspath(video_path),
        native_fps=native_fps,
        sampling_base_fps=sampling_base_fps,
        num_source_frames=int(len(reader)),
        sampled_frame_indices=frame_indices,
        sampled_timestamps_sec=timestamps_sec,
        _reader=reader,
    )


def _sample_imageio_video(video_path: str, sample_fps: float) -> SampledVideo:
    import imageio.v2 as imageio

    reader = imageio.get_reader(video_path)
    metadata = reader.get_meta_data()
    native_fps = float(metadata.get("fps") or 0.0)
    if native_fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {video_path}")

    try:
        num_source_frames = int(reader.count_frames())
    except Exception:
        duration = metadata.get("duration")
        if duration is None or float(duration) <= 0:
            raise ValueError(f"Could not determine frame count for video: {video_path}")
        num_source_frames = max(int(round(float(duration) * native_fps)), 1)

    sampling_base_fps = max(int(round(native_fps)), 1)
    stride = max(int(sampling_base_fps / sample_fps), 1)
    frame_indices = list(range(0, num_source_frames, stride))
    timestamps_sec = [frame_index / sampling_base_fps for frame_index in frame_indices]

    return SampledVideo(
        video_path=os.path.abspath(video_path),
        native_fps=native_fps,
        sampling_base_fps=sampling_base_fps,
        num_source_frames=num_source_frames,
        sampled_frame_indices=frame_indices,
        sampled_timestamps_sec=timestamps_sec,
        _reader=reader,
    )


def sample_video_frames(
    video_path: str,
    sample_fps: float,
    *,
    duration_sec: float | None = None,
    decode_threads: int = 1,
) -> SampledVideo:
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be > 0, got {sample_fps}")

    suffix = Path(video_path).suffix.lower()
    if suffix == ".npy":
        return _sample_numpy_video(video_path, sample_fps, duration_sec)
    return _sample_decord_video(video_path, sample_fps, decode_threads=decode_threads)
