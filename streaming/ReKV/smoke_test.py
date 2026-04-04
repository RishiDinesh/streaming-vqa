#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .datasets import RVSEgoDataset, RVSMovieDataset, build_dataset_from_args, sample_video_frames
from .methods import MethodAnswer, StreamingMethod
from .plot_results import display_label
from .run_eval import evaluate_samples


class RecordingMethod(StreamingMethod):
    def __init__(self, method_name: str) -> None:
        self.method_name = method_name
        self.reset_calls = 0
        self.answer_calls = 0
        self.frames_ingested = 0
        self.ingested_timestamps_sec: list[float] = []
        self.ingest_latencies_sec: list[float] = []

    def reset(self, sample_metadata: dict) -> None:
        self.reset_calls += 1
        self.frames_ingested = 0
        self.ingested_timestamps_sec = []
        self.ingest_latencies_sec = []

    def ingest_frame(self, frame: np.ndarray, timestamp_sec: float) -> dict:
        self.frames_ingested += 1
        self.ingested_timestamps_sec.append(float(timestamp_sec))
        self.ingest_latencies_sec.append(0.001)
        return {
            "timestamp_sec": float(timestamp_sec),
            "ingest_latency_sec": 0.001,
            "frame_token_count": 196,
        }

    def answer_question(self, question: str, metadata: dict | None = None) -> MethodAnswer:
        self.answer_calls += 1
        return MethodAnswer(
            prediction=f"stub-answer-{self.answer_calls}",
            stats={
                "method_name": self.method_name,
                "ttft_sec": 0.01,
                "answer_latency_sec": 0.02,
                "peak_memory_bytes": None,
                "frames_ingested_so_far": self.frames_ingested,
            },
        )

    def get_runtime_stats(self) -> dict:
        return {
            "method_name": self.method_name,
            "frames_ingested": self.frames_ingested,
            "avg_frame_ingest_latency_sec": (
                sum(self.ingest_latencies_sec) / len(self.ingest_latencies_sec)
                if self.ingest_latencies_sec
                else None
            ),
        }


def create_toy_video(video_path: Path, fps: float = 4.0, num_frames: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (32, 32),
    )
    for idx in range(num_frames):
        frame = np.full((32, 32, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="streaming_rekv_smoke_") as tmp_dir:
        root = Path(tmp_dir)
        video_path = root / "toy.mp4"
        second_video_path = root / "toy_second.mp4"
        movie_video_path = root / "toy_movie.npy"
        create_toy_video(video_path)
        create_toy_video(second_video_path)
        np.save(movie_video_path, np.stack([np.full((24, 40, 3), idx, dtype=np.uint8) for idx in range(6)]))

        annotation_path = root / "ego4d_oe.json"
        annotation_path.write_text(
            json.dumps(
                [
                    {
                        "video_id": "toy-video",
                        "video_path": str(video_path),
                        "duration": 2.0,
                        "conversations": [
                            {
                                "question": "Q2",
                                "answer": "A2",
                                "start_time": 0.5,
                                "end_time": 1.5,
                            },
                            {
                                "question": "Q1",
                                "answer": "A1",
                                "start_time": 0.0,
                                "end_time": 0.5,
                            },
                        ],
                    },
                    {
                        "video_id": "toy-video-2",
                        "video_path": str(second_video_path),
                        "duration": 2.0,
                        "conversations": [
                            {
                                "question": "Q3",
                                "answer": "A3",
                                "start_time": 0.25,
                                "end_time": 1.0,
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )

        dataset = RVSEgoDataset(annotation_path=str(annotation_path), video_root=str(root))
        samples = dataset.load()
        assert len(samples) == 2
        assert [conv.question for conv in samples[0].conversations] == ["Q1", "Q2"]
        offset_samples = dataset.load(video_offset=1, max_videos=1)
        assert len(offset_samples) == 1
        assert offset_samples[0].video_id == "toy-video-2"
        repeat_offset_samples = dataset.load(video_offset=1, max_videos=1)
        assert [sample.video_id for sample in offset_samples] == [
            sample.video_id for sample in repeat_offset_samples
        ]

        movie_annotation_path = root / "movienet_oe.json"
        movie_annotation_path.write_text(
            json.dumps(
                [
                    {
                        "video_id": "toy-movie",
                        "video_path": str(movie_video_path),
                        "duration": 6.0,
                        "clip_start_time": 1.0,
                        "clip_end_time": 5.0,
                        "conversations": [
                            {
                                "question": "MQ1",
                                "answer": "MA1",
                                "start_time": 1.0,
                                "end_time": 3.0,
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        movie_dataset = RVSMovieDataset(annotation_path=str(movie_annotation_path), video_root=str(root))
        movie_samples = movie_dataset.load()
        assert len(movie_samples) == 1
        sampled_movie = sample_video_frames(movie_samples[0].video_path, 0.5, duration_sec=movie_samples[0].duration)
        assert sampled_movie.sampling_base_fps == 1
        assert sampled_movie.sampled_timestamps_sec == [0.0, 2.0, 4.0]
        assert sampled_movie.get_frame(0).shape == (24, 40, 3)

        dataset_args = type(
            "Args",
            (),
            {
                "dataset": "rvs_movie",
                "annotation_path": str(movie_annotation_path),
                "video_root": str(root),
                "hf_repo_id": "Becomebright/RVS",
                "allow_hf_video_download": False,
            },
        )()
        assert build_dataset_from_args(dataset_args).dataset_name == "rvs_movie"

        run_config = {"dataset": "rvs_ego", "method": "recording"}
        left = RecordingMethod("recording_left")
        right = RecordingMethod("recording_right")
        left_result = evaluate_samples(
            samples=samples[:1],
            method=left,
            sample_fps=2.0,
            run_config=run_config,
        )
        right_result = evaluate_samples(
            samples=samples[:1],
            method=right,
            sample_fps=2.0,
            run_config=run_config,
            show_progress_bar=False,
        )
        resumed = RecordingMethod("recording_resumed")
        resumed_result = evaluate_samples(
            samples=samples,
            method=resumed,
            sample_fps=2.0,
            run_config=run_config,
            existing_videos=left_result["videos"],
            total_requested_videos=len(samples),
            started_at_utc="2026-04-04T00:00:00+00:00",
            show_progress_bar=False,
        )

        assert left.reset_calls == 1
        assert left.answer_calls == 2
        assert left_result["videos"][0]["conversations"][0]["new_frame_timestamps_sec"] == [0.0]
        assert left_result["videos"][0]["conversations"][1]["new_frame_timestamps_sec"] == [0.5, 1.0]
        assert left_result["aggregate_metrics"]["primary_quality_metric"] == "avg_rouge_l_f1"
        assert left_result["aggregate_metrics"]["primary_quality_score"] is not None
        assert "scores" in left_result["videos"][0]["conversations"][0]
        assert "rouge_l_f1" in left_result["videos"][0]["conversations"][0]["scores"]

        for conversation in left_result["videos"][0]["conversations"]:
            assert all(
                timestamp < conversation["end_time"]
                for timestamp in conversation["sampled_timestamps_sec_so_far"]
            )

        left_timestamps = [
            item["new_frame_timestamps_sec"]
            for item in left_result["videos"][0]["conversations"]
        ]
        right_timestamps = [
            item["new_frame_timestamps_sec"]
            for item in right_result["videos"][0]["conversations"]
        ]
        assert left_timestamps == right_timestamps
        assert resumed.reset_calls == 1
        assert resumed.answer_calls == 1
        assert len(resumed_result["videos"]) == 2
        assert resumed_result["videos"][0]["video_id"] == "toy-video"
        assert resumed_result["videos"][1]["video_id"] == "toy-video-2"
        assert resumed_result["run_state"]["completed_videos"] == 2
        assert resumed_result["run_state"]["total_requested_videos"] == 2

        assert display_label({"run_config": {"method": "full_streaming", "sparsity": 0.5}}) == (
            "full_streaming"
        )
        assert display_label({"run_config": {"method": "duo_streaming", "sparsity": 0.5}}) == (
            "duo_streaming (s=0.5)"
        )
        assert display_label({"run_config": {"method": "duo_streaming", "sparsity": 0.0}}) == (
            "duo_streaming (s=0.0)"
        )
        assert display_label(
            {"run_config": {"method": "rekv", "retrieve_size": 64, "n_local": 15000}}
        ) == "rekv (topk=64,n_local=15000)"
        assert display_label(
            {"run_config": {"method": "duo_plus_rekv", "retrieve_size": 64, "sparsity": 0.5}}
        ) == "duo_plus_rekv (topk=64,s=0.5)"

        print("streaming/ReKV smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
