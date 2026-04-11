#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datasets import RVSEgoDataset, RVSMovieDataset, build_dataset_from_args, sample_video_frames
from .plot_results import (
    display_label,
    plot_cpu_offload_comparison,
    plot_efficiency_vs_context,
    plot_rekv_retrieval,
    plot_retrieval_timeline,
)
from .run_eval import conversation_target_frame_count, evaluate_samples


@dataclass(frozen=True)
class MethodAnswer:
    prediction: str
    stats: dict


class RecordingMethod:
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

    def ingest_features(self, feature_tensor: np.ndarray, timestamp_sec: float) -> dict:
        return self.ingest_frame(np.asarray(feature_tensor), timestamp_sec)

    def answer_question(self, question: str, metadata: dict | None = None) -> MethodAnswer:
        self.answer_calls += 1
        return MethodAnswer(
            prediction=f"stub-answer-{question}",
            stats={
                "method_name": self.method_name,
                "ttft_sec": 0.01,
                "answer_latency_sec": 0.02,
                "peak_memory_bytes": None,
                "cpu_offload_bytes_current": 0,
                "cpu_offload_bytes_peak": 0,
                "retrieval_latency_sec": None,
                "avg_retrieved_block_count": 0.0,
                "retrieved_block_indices_union": [],
                "retrieved_timestamps_sec_union": [],
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
            "cumulative_frame_ingest_latency_sec": sum(self.ingest_latencies_sec),
        }


def create_toy_numpy_video(video_path: Path, num_frames: int = 8, height: int = 32, width: int = 32) -> None:
    np.save(
        video_path,
        np.stack(
            [np.full((height, width, 3), idx * 20, dtype=np.uint8) for idx in range(num_frames)],
            axis=0,
        ),
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="streaming_rekv_smoke_") as tmp_dir:
        root = Path(tmp_dir)
        video_path = root / "toy.npy"
        second_video_path = root / "toy_second.npy"
        movie_video_path = root / "toy_movie.npy"
        create_toy_numpy_video(video_path)
        create_toy_numpy_video(second_video_path)
        np.save(movie_video_path, np.stack([np.full((24, 40, 3), idx, dtype=np.uint8) for idx in range(6)]))

        annotation_path = root / "ego4d_oe.json"
        annotation_path.write_text(
            json.dumps(
                [
                    {
                        "video_id": "toy-video",
                        "video_path": str(video_path),
                        "duration": 2.0,
                        "scene_type": "ego_outdoor",
                        "conversations": [
                            {
                                "question": "Q2",
                                "answer": "A2",
                                "start_time": 0.5,
                                "end_time": 1.5,
                                "qid": "ego-q2",
                            },
                            {
                                "question": "Q1",
                                "answer": "A1",
                                "start_time": 0.0,
                                "end_time": 0.5,
                                "qid": "ego-q1",
                            },
                        ],
                    },
                    {
                        "video_id": "toy-video-2",
                        "video_path": str(second_video_path),
                        "duration": 2.0,
                        "scene_type": "ego_indoor",
                        "conversations": [
                            {
                                "question": "Q3",
                                "answer": "A3",
                                "start_time": 0.25,
                                "end_time": 1.0,
                                "qid": "ego-q3",
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
        evaluation_manifest = {
            "comparison_contract_version": "v1",
            "comparison_scope": "streaming_vqa_cross_method",
            "method_manifest": {"method_name": "recording"},
        }
        left = RecordingMethod("recording_left")
        right = RecordingMethod("recording_right")
        left_result = evaluate_samples(
            samples=samples[:1],
            method=left,
            sample_fps=2.0,
            run_config=run_config,
            evaluation_manifest=evaluation_manifest,
        )
        right_result = evaluate_samples(
            samples=samples[:1],
            method=right,
            sample_fps=2.0,
            run_config=run_config,
            evaluation_manifest=evaluation_manifest,
            show_progress_bar=False,
        )
        resumed = RecordingMethod("recording_resumed")
        resumed_result = evaluate_samples(
            samples=samples,
            method=resumed,
            sample_fps=2.0,
            run_config=run_config,
            evaluation_manifest=evaluation_manifest,
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
        assert left_result["evaluation_manifest"] == evaluation_manifest
        assert "scores" in left_result["videos"][0]["conversations"][0]
        assert "rouge_l_f1" in left_result["videos"][0]["conversations"][0]["scores"]
        assert left_result["videos"][0]["extra_metadata"] == {"scene_type": "ego_outdoor"}
        assert left_result["videos"][0]["conversations"][0]["extra_metadata"] == {"qid": "ego-q1"}
        assert conversation_target_frame_count(0.5, [0.0, 0.5, 1.0, 1.5]) == 1
        assert conversation_target_frame_count(0.75, [0.0, 0.5, 1.0, 1.5]) == 2
        assert conversation_target_frame_count(1.0, [0.0, 0.5, 1.0, 1.5]) == 2
        assert conversation_target_frame_count(1.01, [0.0, 0.5, 1.0, 1.5]) == 3

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
        assert resumed_result["in_progress_video"] is None

        partial_video = copy.deepcopy(left_result["videos"][0])
        partial_video["conversations"] = partial_video["conversations"][:1]
        partial_video["runtime_stats"] = {
            "method_name": "recording_partial",
            "frames_ingested": 1,
            "avg_frame_ingest_latency_sec": 0.001,
            "cumulative_frame_ingest_latency_sec": 0.001,
            "last_ingested_timestamp_sec": 0.0,
        }
        partial_video["checkpoint_state"] = {
            "completed_conversations": 1,
            "frames_ingested": 1,
            "ingested_until_frame_index": 0,
            "last_ingested_timestamp_sec": 0.0,
        }
        partial_resumed = RecordingMethod("recording_partial_resumed")
        partial_resumed_result = evaluate_samples(
            samples=samples[:1],
            method=partial_resumed,
            sample_fps=2.0,
            run_config=run_config,
            evaluation_manifest=evaluation_manifest,
            total_requested_videos=1,
            started_at_utc="2026-04-04T00:00:00+00:00",
            show_progress_bar=False,
            existing_in_progress_video=partial_video,
        )
        assert partial_resumed.reset_calls == 1
        assert partial_resumed.answer_calls == 1
        assert partial_resumed_result["run_state"]["completed_videos"] == 1
        assert partial_resumed_result["in_progress_video"] is None
        assert (
            partial_resumed_result["videos"][0]["conversations"][0]["prediction"]
            == left_result["videos"][0]["conversations"][0]["prediction"]
        )
        assert (
            partial_resumed_result["videos"][0]["conversations"][1]["prediction"]
            == left_result["videos"][0]["conversations"][1]["prediction"]
        )
        assert abs(
            partial_resumed_result["videos"][0]["runtime_stats"]["cumulative_frame_ingest_latency_sec"]
            - 0.003
        ) < 1e-9

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
            {"run_config": {"method": "rekv_no_offload", "n_local": 15000}}
        ) == "rekv_no_offload (n_local=15000)"
        assert display_label(
            {"run_config": {"method": "duo_plus_rekv", "retrieve_size": 64, "sparsity": 0.375}}
        ) == "duo_plus_rekv (topk=64,s=0.375)"

        rekv_payload = copy.deepcopy(left_result)
        rekv_payload["run_config"] = {"method": "rekv", "retrieve_size": 64, "n_local": 15000}
        for conversation in rekv_payload["videos"][0]["conversations"]:
            conversation["method_stats"].update(
                {
                    "peak_memory_bytes": 1024**3,
                    "cpu_offload_bytes_current": 256 * 1024**2,
                    "cpu_offload_bytes_peak": 512 * 1024**2,
                    "retrieval_latency_sec": 0.03,
                    "avg_retrieved_block_count": 4.0,
                    "retrieved_block_indices_union": [0, 1],
                    "retrieved_timestamps_sec_union": [0.0, 0.5],
                }
            )
        rekv_payload["aggregate_metrics"]["peak_cpu_offload_bytes"] = 512 * 1024**2

        no_offload_payload = copy.deepcopy(left_result)
        no_offload_payload["run_config"] = {"method": "rekv_no_offload", "n_local": 15000}
        for conversation in no_offload_payload["videos"][0]["conversations"]:
            conversation["method_stats"].update(
                {
                    "peak_memory_bytes": 900 * 1024**2,
                    "cpu_offload_bytes_current": 0,
                    "cpu_offload_bytes_peak": 0,
                    "retrieval_latency_sec": None,
                    "avg_retrieved_block_count": 0.0,
                    "retrieved_block_indices_union": [],
                    "retrieved_timestamps_sec_union": [],
                }
            )
        no_offload_payload["aggregate_metrics"]["peak_cpu_offload_bytes"] = 0

        hybrid_payload = copy.deepcopy(left_result)
        hybrid_payload["run_config"] = {
            "method": "duo_plus_rekv",
            "retrieve_size": 64,
            "sparsity": 0.5,
        }
        for conversation in hybrid_payload["videos"][0]["conversations"]:
            conversation["method_stats"].update(
                {
                    "peak_memory_bytes": 1100 * 1024**2,
                    "cpu_offload_bytes_current": 128 * 1024**2,
                    "cpu_offload_bytes_peak": 256 * 1024**2,
                    "retrieval_latency_sec": 0.02,
                    "avg_retrieved_block_count": 3.0,
                    "retrieved_block_indices_union": [1],
                    "retrieved_timestamps_sec_union": [0.5],
                }
            )
        hybrid_payload["aggregate_metrics"]["peak_cpu_offload_bytes"] = 256 * 1024**2

        plot_dir = root / "smoke_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        cpu_plot = plot_cpu_offload_comparison(
            [rekv_payload, no_offload_payload, hybrid_payload],
            plot_dir,
        )
        assert cpu_plot is not None and cpu_plot.is_file()
        assert plot_efficiency_vs_context(
            [rekv_payload, no_offload_payload, hybrid_payload],
            plot_dir,
        ).is_file()
        assert plot_rekv_retrieval([rekv_payload, no_offload_payload, hybrid_payload], plot_dir).is_file()
        assert plot_retrieval_timeline([rekv_payload, hybrid_payload], plot_dir).is_file()

        print("streaming/ReKV smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
