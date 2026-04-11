from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


FEATURE_CACHE_VERSION = "v1"


def slugify(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")


def default_feature_cache_root(dataset: str, model: str, sample_fps: float) -> Path:
    model_slug = slugify(model)
    fps_slug = str(sample_fps).replace(".", "p")
    return (
        Path("outputs")
        / "evaluations_streaming"
        / "feature_cache"
        / dataset.replace("_", "-")
        / model_slug
        / f"fps_{fps_slug}"
    )


def feature_cache_videos_dir(cache_root: Path) -> Path:
    return cache_root / "videos"


def feature_cache_file_name(sample_id: str) -> str:
    return f"{slugify(sample_id)}.pt"


def feature_cache_path(cache_root: Path, sample_id: str) -> Path:
    return feature_cache_videos_dir(cache_root) / feature_cache_file_name(sample_id)


def feature_cache_manifest_path(cache_root: Path) -> Path:
    return cache_root / "manifest.json"


def compute_expected_sampling_schedule(
    *,
    num_source_frames: int,
    sampling_base_fps: int,
    sample_fps: float,
) -> tuple[list[int], list[float]]:
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be > 0, got {sample_fps}")
    if sampling_base_fps <= 0:
        raise ValueError(f"sampling_base_fps must be > 0, got {sampling_base_fps}")
    stride = max(int(sampling_base_fps / sample_fps), 1)
    frame_indices = list(range(0, num_source_frames, stride))
    timestamps_sec = [frame_index / sampling_base_fps for frame_index in frame_indices]
    return frame_indices, timestamps_sec


@dataclass(frozen=True)
class CachedFeatureVideo:
    sample_id: str
    video_id: str
    video_path: str
    duration: float
    sample_fps: float
    native_fps: float
    sampling_base_fps: int
    num_source_frames: int
    sampled_frame_indices: list[int]
    sampled_timestamps_sec: list[float]
    features: torch.Tensor
    cache_path: str

    def get_feature(self, sampled_index: int) -> torch.Tensor:
        return self.features[sampled_index]


def load_feature_cache_manifest(cache_root: Path) -> dict[str, Any]:
    manifest_path = feature_cache_manifest_path(cache_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Feature cache manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_feature_cache_manifest(cache_root: Path, payload: dict[str, Any]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = feature_cache_manifest_path(cache_root)
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(manifest_path)


def validate_feature_cache_payload(
    payload: dict[str, Any],
    *,
    sample_id: str,
    video_id: str,
    sample_fps: float,
) -> None:
    if str(payload.get("sample_id")) != str(sample_id):
        raise ValueError(
            f"Feature cache sample_id mismatch: expected {sample_id!r}, got {payload.get('sample_id')!r}"
        )
    if str(payload.get("video_id")) != str(video_id):
        raise ValueError(
            f"Feature cache video_id mismatch: expected {video_id!r}, got {payload.get('video_id')!r}"
        )

    payload_sample_fps = float(payload.get("sample_fps"))
    if abs(payload_sample_fps - float(sample_fps)) > 1e-9:
        raise ValueError(
            f"Feature cache sample_fps mismatch: expected {sample_fps}, got {payload_sample_fps}"
        )

    expected_indices, expected_timestamps = compute_expected_sampling_schedule(
        num_source_frames=int(payload["num_source_frames"]),
        sampling_base_fps=int(payload["sampling_base_fps"]),
        sample_fps=float(sample_fps),
    )
    cached_indices = [int(value) for value in payload["sampled_frame_indices"]]
    cached_timestamps = [float(value) for value in payload["sampled_timestamps_sec"]]
    if cached_indices != expected_indices:
        raise ValueError(
            "Feature cache sampled_frame_indices do not match the expected schedule for "
            f"{sample_id}: expected {len(expected_indices)} indices, got {len(cached_indices)}"
        )
    if len(cached_timestamps) != len(expected_timestamps):
        raise ValueError(
            "Feature cache sampled_timestamps_sec length mismatch for "
            f"{sample_id}: expected {len(expected_timestamps)}, got {len(cached_timestamps)}"
        )
    for expected, actual in zip(expected_timestamps, cached_timestamps):
        if abs(expected - actual) > 1e-6:
            raise ValueError(
                "Feature cache sampled_timestamps_sec do not match the expected schedule for "
                f"{sample_id}: expected {expected}, got {actual}"
            )

    features = payload.get("features")
    if not isinstance(features, torch.Tensor):
        raise TypeError("Feature cache payload is missing a tensor under 'features'.")
    if features.ndim != 3:
        raise ValueError(
            f"Feature cache tensor must have shape [num_frames, num_frame_tokens, hidden_dim], got {tuple(features.shape)}"
        )
    if int(features.shape[0]) != len(cached_indices):
        raise ValueError(
            "Feature cache tensor frame count does not match sampled schedule length: "
            f"{int(features.shape[0])} vs {len(cached_indices)}"
        )


def load_cached_feature_video(
    cache_root: Path,
    *,
    sample_id: str,
    video_id: str,
    sample_fps: float,
) -> CachedFeatureVideo:
    cache_path = feature_cache_path(cache_root, sample_id)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cached feature file not found: {cache_path}")

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Cached feature file must contain a dict payload: {cache_path}")

    validate_feature_cache_payload(
        payload,
        sample_id=sample_id,
        video_id=video_id,
        sample_fps=sample_fps,
    )
    features = payload["features"].to(dtype=torch.bfloat16, device="cpu").contiguous()
    return CachedFeatureVideo(
        sample_id=str(payload["sample_id"]),
        video_id=str(payload["video_id"]),
        video_path=str(payload["video_path"]),
        duration=float(payload["duration"]),
        sample_fps=float(payload["sample_fps"]),
        native_fps=float(payload["native_fps"]),
        sampling_base_fps=int(payload["sampling_base_fps"]),
        num_source_frames=int(payload["num_source_frames"]),
        sampled_frame_indices=[int(value) for value in payload["sampled_frame_indices"]],
        sampled_timestamps_sec=[float(value) for value in payload["sampled_timestamps_sec"]],
        features=features,
        cache_path=str(cache_path),
    )
