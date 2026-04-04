#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .datasets import RVSEgoDataset
from .methods import MethodAnswer, StreamingMethod
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
        create_toy_video(video_path)

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
                    }
                ]
            ),
            encoding="utf-8",
        )

        dataset = RVSEgoDataset(annotation_path=str(annotation_path), video_root=str(root))
        samples = dataset.load()
        assert len(samples) == 1
        assert [conv.question for conv in samples[0].conversations] == ["Q1", "Q2"]

        run_config = {"dataset": "rvs_ego", "method": "recording"}
        left = RecordingMethod("recording_left")
        right = RecordingMethod("recording_right")
        left_result = evaluate_samples(
            samples=samples,
            method=left,
            sample_fps=0.5,
            run_config=run_config,
        )
        right_result = evaluate_samples(
            samples=samples,
            method=right,
            sample_fps=0.5,
            run_config=run_config,
        )

        assert left.reset_calls == 1
        assert left.answer_calls == 2
        assert left_result["videos"][0]["conversations"][0]["new_frame_timestamps_sec"] == [0.0]
        assert left_result["videos"][0]["conversations"][1]["new_frame_timestamps_sec"] == []

        for conversation in left_result["videos"][0]["conversations"]:
            assert all(
                timestamp <= conversation["end_time"]
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

        print("streaming/ReKV smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
