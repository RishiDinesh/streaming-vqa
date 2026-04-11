#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact backend/comparability report from streaming audit outputs."
    )
    parser.add_argument("--result-dir", required=True, help="Directory containing per-method result JSONs.")
    parser.add_argument("--profile-dir", required=True, help="Directory containing per-method profile JSONs.")
    parser.add_argument("--env-summary", required=True, help="Path to the saved ROCm validator JSON.")
    parser.add_argument("--output-path", required=True, help="Markdown report output path.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _maybe_gb(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) / (1024 ** 3):.3f} GB"


def _method_backend(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload.get("evaluation_manifest", {}).get("method_manifest", {}).get("backend_resolution", {}) or {})


def _profile_summary(payload: dict[str, Any]) -> dict[str, Any]:
    probes = payload.get("video_profile", {}).get("probes", [])
    if not probes:
        return {}
    last_probe = probes[-1]
    return {
        "frames": last_probe.get("ingested_frame_count"),
        "answer_latency_sec": last_probe.get("method_stats", {}).get("answer_latency_sec"),
        "ttft_sec": last_probe.get("method_stats", {}).get("ttft_sec"),
        "peak_memory_bytes": last_probe.get("method_stats", {}).get("peak_memory_bytes"),
        "cpu_offload_bytes_current": last_probe.get("method_stats", {}).get("cpu_offload_bytes_current"),
        "retrieval_latency_sec": last_probe.get("method_stats", {}).get("retrieval_latency_sec"),
        "avg_retrieved_block_count": last_probe.get("method_stats", {}).get("avg_retrieved_block_count"),
    }


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir).expanduser().resolve()
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    env_summary = _load_json(Path(args.env_summary).expanduser().resolve())

    result_payloads = {}
    for path in sorted(result_dir.glob("*.json")):
        payload = _load_json(path)
        method = payload.get("run_config", {}).get("method")
        if method:
            result_payloads[method] = payload

    profile_payloads = {}
    for path in sorted(profile_dir.glob("*.json")):
        payload = _load_json(path)
        method = payload.get("run_config", {}).get("method")
        if method:
            profile_payloads[method] = payload

    methods = ["full_streaming", "duo_streaming", "rekv", "duo_plus_rekv"]
    lines: list[str] = []
    lines.append("# ROCm Backend Audit Report")
    lines.append("")
    lines.append("## Environment")
    lines.append(f"- Python: `{env_summary.get('python', {}).get('executable')}`")
    lines.append(
        f"- Torch/ROCm: `{env_summary.get('torch_runtime', {}).get('torch_version')}` / "
        f"`{env_summary.get('torch_runtime', {}).get('hip_version')}`"
    )
    lines.append("")
    lines.append("## Method Comparison")
    lines.append("")
    lines.append("| Method | Quality | TTFT | Answer Latency | Peak GPU | CPU Offload | Retrieval | Duo Streaming Backend | Fallback | Category |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")

    for method in methods:
        result = result_payloads.get(method, {})
        aggregate = result.get("aggregate_metrics", {})
        backend = _method_backend(result)
        profile = _profile_summary(profile_payloads.get(method, {}))
        quality = aggregate.get("primary_quality_score")
        duo_backend = backend.get("streaming_attn_backend_actual") or "-"
        fallback = backend.get("streaming_attn_fallback_reason") or "-"
        category = backend.get("result_interpretation_category") or "-"
        lines.append(
            "| {method} | {quality} | {ttft} | {lat} | {peak} | {cpu} | {retrieval} | {duo_backend} | {fallback} | {category} |".format(
                method=method,
                quality=f"{quality:.4f}" if isinstance(quality, (int, float)) else "-",
                ttft=(
                    f"{aggregate.get('avg_ttft_sec'):.4f}s"
                    if isinstance(aggregate.get("avg_ttft_sec"), (int, float))
                    else "-"
                ),
                lat=(
                    f"{aggregate.get('avg_answer_latency_sec'):.4f}s"
                    if isinstance(aggregate.get("avg_answer_latency_sec"), (int, float))
                    else "-"
                ),
                peak=_maybe_gb(aggregate.get("peak_memory_bytes")),
                cpu=_maybe_gb(aggregate.get("peak_cpu_offload_bytes")),
                retrieval=(
                    f"{aggregate.get('avg_retrieval_latency_sec'):.4f}s / "
                    f"{aggregate.get('avg_retrieved_block_count'):.2f}"
                    if isinstance(aggregate.get("avg_retrieval_latency_sec"), (int, float))
                    else "-"
                ),
                duo_backend=duo_backend,
                fallback=fallback,
                category=category,
            )
        )
        if profile:
            lines.append(
                f"  profile tail: frames={profile.get('frames')} ttft={profile.get('ttft_sec')} "
                f"answer_latency={profile.get('answer_latency_sec')} retrieval={profile.get('retrieval_latency_sec')}"
            )
    lines.append("")
    lines.append("## Comparability Checks")
    for method in methods:
        result = result_payloads.get(method, {})
        manifest = result.get("evaluation_manifest", {})
        shared = manifest.get("shared_run_settings", {})
        protocol = manifest.get("streaming_protocol", {})
        if not shared:
            continue
        lines.append(
            f"- `{method}`: sample_fps={shared.get('sample_fps')} "
            f"ingest_source={shared.get('ingest_source')} "
            f"cutoff={protocol.get('causal_cutoff_policy')}"
        )

    output_path = Path(args.output_path).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved backend audit report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
