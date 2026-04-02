#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LMMS_ROOT = REPO_ROOT / "lmms-eval"


@dataclass(frozen=True)
class LaunchConfig:
    num_processes: int
    num_machines: int
    machine_rank: int | None
    main_process_ip: str | None
    main_process_port: int | None
    same_network: bool

def _load_duo_attention_eval_utils():
    module_path = LMMS_ROOT / "lmms_eval" / "models" / "model_utils" / "duo_attention_eval.py"
    spec = importlib.util.spec_from_file_location("duo_attention_eval_for_launcher", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DuoAttention helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_duo_eval_utils = _load_duo_attention_eval_utils()
extract_experiment_name = _duo_eval_utils.extract_experiment_name
normalize_attention_dir = _duo_eval_utils.normalize_attention_dir


def optional_float_arg(value: str):
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    return float(value)


def resolve_mode(mode: str, attn_dir: str | None) -> str:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"auto", "baseline", "duo"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if normalized_mode == "auto":
        return "duo" if attn_dir else "baseline"
    if normalized_mode == "duo" and not attn_dir:
        raise ValueError("--attn-dir is required when --mode=duo")
    return normalized_mode


def pretrained_slug(pretrained: str) -> str:
    candidate = Path(pretrained.rstrip("/")).name or pretrained.strip().split("/")[-1]
    return candidate.replace(" ", "_")


def resolve_output_stem(pretrained: str, mode: str, normalized_attn_dir: Path | None, output_name: str | None) -> str:
    if output_name:
        return output_name
    if mode == "duo":
        if normalized_attn_dir is None:
            raise ValueError("normalized_attn_dir is required for DuoAttention output naming")
        return f"{extract_experiment_name(normalized_attn_dir)}_eval"
    return f"{pretrained_slug(pretrained)}_baseline_eval"


def build_model_args(args, normalized_attn_dir: Path | None, mode: str) -> str:
    model_args = [
        f"pretrained={args.pretrained}",
        f"fps={args.fps}",
        f"video_decode_backend={args.video_decode_backend}",
        f"max_frames_num={args.max_frames_num}",
        f"cap_fps_sampling={args.cap_fps_sampling}",
        f"decoding_simulation_length={args.decoding_simulation_length}",
    ]
    if args.device_map is not None and args.device_map.strip():
        model_args.append(f"device_map={args.device_map}")
    if args.deploy_sink_size is not None:
        model_args.append(f"deploy_sink_size={args.deploy_sink_size}")
    if args.deploy_recent_size is not None:
        model_args.append(f"deploy_recent_size={args.deploy_recent_size}")
    if mode == "duo":
        if normalized_attn_dir is None:
            raise ValueError("normalized_attn_dir is required when building DuoAttention model args")
        model_args.append(f"attn_dir={normalized_attn_dir}")
        if args.sparsity is None:
            model_args.append("sparsity=none")
        else:
            model_args.append(f"sparsity={args.sparsity}")
        if args.threshold is not None:
            model_args.append(f"threshold={args.threshold}")
    return ",".join(model_args)


def _resolve_int_env(*names: str) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value is None or not value.strip():
            continue
        return int(value)
    return None


def resolve_launch_config(args) -> LaunchConfig:
    inferred_num_machines = _resolve_int_env("NUM_MACHINES", "SLURM_NNODES", "SLURM_JOB_NUM_NODES")
    num_machines = args.num_machines if args.num_machines is not None else (inferred_num_machines or 1)

    machine_rank = args.machine_rank
    if machine_rank is None and num_machines > 1:
        machine_rank = _resolve_int_env("MACHINE_RANK", "SLURM_PROCID", "SLURM_NODEID", "NODE_RANK", "RANK")

    main_process_ip = args.main_process_ip
    if main_process_ip is None and num_machines > 1:
        main_process_ip = os.environ.get("MAIN_PROCESS_IP") or os.environ.get("MASTER_ADDR")

    main_process_port = args.main_process_port
    if main_process_port is None and num_machines > 1:
        main_process_port = _resolve_int_env("MAIN_PROCESS_PORT", "MASTER_PORT")
        if main_process_port is None:
            main_process_port = 29500

    if args.num_processes == 1 and num_machines > 1:
        num_processes = num_machines
    else:
        num_processes = args.num_processes

    if num_processes < num_machines:
        raise ValueError(
            f"--num-processes ({num_processes}) must be >= --num-machines ({num_machines}) "
            "because Accelerate expects the total process count."
        )

    if num_machines > 1:
        if machine_rank is None:
            raise ValueError(
                "Multi-node launch requires --machine-rank or one of MACHINE_RANK/SLURM_PROCID/SLURM_NODEID in the environment."
            )
        if main_process_ip is None:
            raise ValueError(
                "Multi-node launch requires --main-process-ip or MASTER_ADDR/MAIN_PROCESS_IP in the environment."
            )

    return LaunchConfig(
        num_processes=num_processes,
        num_machines=num_machines,
        machine_rank=machine_rank,
        main_process_ip=main_process_ip,
        main_process_port=main_process_port,
        same_network=args.same_network,
    )


def build_launch_command(
    args,
    output_path: Path,
    model_args: str,
    passthrough_args: list[str],
    launch_config: LaunchConfig,
) -> list[str]:
    eval_args = [
        "--model",
        args.model,
        "--model_args",
        model_args,
        "--tasks",
        args.task,
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        str(output_path),
    ]
    if args.limit is not None:
        eval_args.extend(["--limit", str(args.limit)])
    if args.log_samples:
        eval_args.append("--log_samples")

    passthrough_args = [arg for arg in passthrough_args if arg != "--"]
    eval_args.extend(passthrough_args)

    if launch_config.num_processes == 1 and launch_config.num_machines == 1:
        return [sys.executable, "-m", "lmms_eval", *eval_args]

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes",
        str(launch_config.num_processes),
    ]
    if launch_config.num_machines > 1:
        cmd.extend(
            [
                "--num_machines",
                str(launch_config.num_machines),
                "--machine_rank",
                str(launch_config.machine_rank),
                "--main_process_ip",
                str(launch_config.main_process_ip),
                "--main_process_port",
                str(launch_config.main_process_port),
            ]
        )
        if launch_config.same_network:
            cmd.append("--same_network")

    cmd.extend(
        [
        "-m",
        "lmms_eval",
        *eval_args,
        ]
    )
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch root lmms-eval for baseline or DuoAttention LLaVA-OneVision evaluation.")
    parser.add_argument("--pretrained", required=True, help="Hugging Face model id or local checkpoint for LLaVA-OneVision.")
    parser.add_argument("--attn-dir", default=None, help="Attention-pattern directory or the full path to full_attention_heads.tsv. Required only for DuoAttention runs.")
    parser.add_argument("--task", required=True, help="lmms-eval task name, such as videomme, mlvu_dev, or egoschema_subset.")
    parser.add_argument("--model", default="llava_onevision_duo_attn", help="lmms-eval model id to launch.")
    parser.add_argument("--mode", default="auto", help="One of: auto, baseline, duo. Auto uses DuoAttention when --attn-dir is given, otherwise baseline.")
    parser.add_argument("--sparsity", type=optional_float_arg, default=0.5, help="DuoAttention sparsity. Use 'none' to disable sparsity-based binarization and fall back to threshold.")
    parser.add_argument("--threshold", type=optional_float_arg, default=None, help="Threshold fallback when sparsity is disabled.")
    parser.add_argument("--limit", default=None, help="Optional lmms-eval sample limit.")
    parser.add_argument("--num-processes", type=int, default=1, help="Total number of processes for accelerate launch across all machines. Use 1 to run without accelerate.")
    parser.add_argument("--num-machines", type=int, default=None, help="Number of machines for multi-node accelerate launch. Defaults to SLURM_NNODES/SLURM_JOB_NUM_NODES when available.")
    parser.add_argument("--machine-rank", type=int, default=None, help="Machine rank for multi-node accelerate launch. Defaults to MACHINE_RANK, SLURM_PROCID, or SLURM_NODEID.")
    parser.add_argument("--main-process-ip", default=None, help="Rank-0 machine IP/hostname for multi-node accelerate launch. Defaults to MASTER_ADDR when set.")
    parser.add_argument("--main-process-port", type=int, default=None, help="Rank-0 rendezvous port for multi-node accelerate launch. Defaults to MASTER_PORT or 29500.")
    parser.add_argument("--same-network", action="store_true", help="Forward --same_network to accelerate for multi-node launches on the same cluster network.")
    parser.add_argument("--batch-size", type=int, default=1, help="lmms-eval batch size.")
    parser.add_argument("--fps", default="auto", help="Video sampling FPS. Use 'auto' for 0.5 FPS <=30 minutes and 0.2 FPS above 30 minutes.")
    parser.add_argument("--max-frames-num", type=int, default=32, help="Hard cap on sampled video frames passed to the model.")
    parser.add_argument("--cap-fps-sampling", action="store_true", help="Apply --max-frames-num even when FPS-based sampling is enabled. Disabled by default to match StreamingTOM behavior.")
    parser.add_argument(
        "--decoding-simulation-length",
        type=int,
        default=0,
        help=(
            "For llava_onevision_duo_attn raw-video eval, simulate the last N prompt "
            "tokens through the KV cache before greedy generation. Use this to mirror "
            "the text DuoAttention LongBench/needle evaluation style."
        ),
    )
    parser.add_argument("--video-decode-backend", default="decord", help="Video decode backend passed to the DuoAttention model.")
    parser.add_argument("--device-map", default="", help="Optional model device_map forwarded through lmms-eval model_args.")
    parser.add_argument("--deploy-sink-size", type=int, default=None, help="Optional deploy-time DuoAttention sink size override.")
    parser.add_argument("--deploy-recent-size", type=int, default=None, help="Optional deploy-time DuoAttention recent size override.")
    parser.add_argument("--output-name", default=None, help="Optional explicit output directory stem under outputs/evaluations/.")
    parser.add_argument("--log-samples", action="store_true", help="Forward --log_samples to lmms-eval.")
    args, passthrough_args = parser.parse_known_args()

    if args.num_processes < 1:
        raise ValueError("--num-processes must be >= 1")
    if args.num_machines is not None and args.num_machines < 1:
        raise ValueError("--num-machines must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_frames_num < 1:
        raise ValueError("--max-frames-num must be >= 1")

    mode = resolve_mode(args.mode, args.attn_dir)
    normalized_attn_dir = normalize_attention_dir(args.attn_dir) if mode == "duo" else None
    output_stem = resolve_output_stem(args.pretrained, mode, normalized_attn_dir, args.output_name)
    output_path = REPO_ROOT / "outputs" / "evaluations" / output_stem
    output_path.mkdir(parents=True, exist_ok=True)

    model_args = build_model_args(args, normalized_attn_dir, mode)
    launch_config = resolve_launch_config(args)

    env = os.environ.copy()
    pythonpath_entries = [str(LMMS_ROOT), str(REPO_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    cmd = build_launch_command(args, output_path, model_args, passthrough_args, launch_config)
    print("Launching:")
    print(" ".join(shlex.quote(part) for part in cmd))
    print(f"Mode: {mode}")
    print(f"Outputs: {output_path}")
    if launch_config.num_machines > 1:
        print(
            "Distributed launch: "
            f"num_processes={launch_config.num_processes}, "
            f"num_machines={launch_config.num_machines}, "
            f"machine_rank={launch_config.machine_rank}, "
            f"main_process={launch_config.main_process_ip}:{launch_config.main_process_port}"
        )

    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
