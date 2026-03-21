import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoProcessor

from duo_attn.data import create_video_qa_dataloader
from duo_attn.utils import load_attn_pattern, seed_everything, sparsify_attention_heads

try:
    from .inference import evaluate_mask_with_fresh_model
    from .reporting import (
        build_report,
        sort_detailed_iterations_for_output,
        sort_rows_for_output,
        write_outputs,
    )
    from .runtime import (
        cleanup_distributed,
        ensure_default_output_paths,
        init_distributed,
        parse_args,
        resolve_device_and_dtype,
    )
    from .sweep import (
        ablate_pool_ratio,
        build_fixed_ratios,
        build_pool,
        deterministic_pool_order,
        mask_cache_key,
    )
except ImportError:
    from inference import evaluate_mask_with_fresh_model
    from reporting import (
        build_report,
        sort_detailed_iterations_for_output,
        sort_rows_for_output,
        write_outputs,
    )
    from runtime import (
        cleanup_distributed,
        ensure_default_output_paths,
        init_distributed,
        parse_args,
        resolve_device_and_dtype,
    )
    from sweep import (
        ablate_pool_ratio,
        build_fixed_ratios,
        build_pool,
        deterministic_pool_order,
        mask_cache_key,
    )


def main() -> None:
    args = parse_args()
    ensure_default_output_paths(args)
    if args.batch_size != 1:
        raise ValueError(
            "--batch_size > 1 is disabled for correctness: patched reordered Qwen2 "
            "eval attention does not apply attention_mask, so padding in batched "
            "inputs can affect results. Use --batch_size 1."
        )
    seed_everything(args.seed)

    device, dtype = resolve_device_and_dtype(args)
    use_distributed, rank, world_size, local_rank = init_distributed(device)
    is_main_process = rank == 0

    def log(msg: str) -> None:
        if (not use_distributed) or is_main_process:
            print(msg)

    if use_distributed:
        log(
            f"Running distributed validation with world_size={world_size} "
            f"(rank={rank}, local_rank={local_rank})"
        )
    log(f"Using device={device}, dtype={dtype}")
    log("Using default DuoAttention eval attention kernels.")

    ratios = build_fixed_ratios()

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if "llava_onevision" not in config.model_type:
        raise ValueError(
            f"This runner expects a Llava-OneVision model, got model_type={config.model_type}."
        )

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    dataloader = create_video_qa_dataloader(
        video_root=args.video_root,
        dataset_name="vnbench",
        annotation_path=args.annotation_path,
        processor=processor,
        model_id=args.model_name,
        num_frames=args.num_frames,
        max_length=args.max_length,
        use_chat_template=not args.disable_video_chat_template,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    full_attention_heads, sink_size, recent_size = load_attn_pattern(args.attn_load_dir)
    learned_mask, actual_sparsity = sparsify_attention_heads(
        full_attention_heads,
        threshold=args.threshold,
        sparsity=args.sparsity,
    )
    learned_mask = learned_mask.astype(float)

    num_layers, num_heads = learned_mask.shape
    full_mask = np.ones_like(learned_mask, dtype=float)

    streaming_pool = build_pool(learned_mask, select_value=0.0)
    retrieval_pool = build_pool(learned_mask, select_value=1.0)

    ordered_streaming_pool = deterministic_pool_order(
        streaming_pool,
        seed=args.seed,
        seed_offset=17,
    )
    ordered_retrieval_pool = deterministic_pool_order(
        retrieval_pool,
        seed=args.seed,
        seed_offset=29,
    )

    warnings: List[str] = []
    if len(streaming_pool) == 0:
        warnings.append(
            "streaming_pool is empty after binarization; sweep will keep all heads full."
        )
    if len(retrieval_pool) == 0:
        warnings.append(
            "retrieval_pool is empty after binarization; sweep will keep all heads full."
        )

    log(
        "Loaded learned mask "
        f"shape=({num_layers}, {num_heads}), actual_sparsity={actual_sparsity:.4f}, "
        f"sink_size={sink_size}, recent_size={recent_size}"
    )
    log(
        "Pool sizes: "
        f"streaming_pool={len(streaming_pool)}, retrieval_pool={len(retrieval_pool)}"
    )
    log(f"Ratio schedule: {ratios}")

    eval_cache: Dict[str, Tuple[float, int, int, List[Dict[str, Any]]]] = {}

    def eval_with_cache(mask: np.ndarray) -> Tuple[float, int, int, List[Dict[str, Any]]]:
        key = mask_cache_key(mask)
        if key in eval_cache:
            log("Reusing cached evaluation for equivalent mask.")
            return eval_cache[key]
        result = evaluate_mask_with_fresh_model(
            args=args,
            config=config,
            processor=processor,
            dataloader=dataloader,
            device=device,
            dtype=dtype,
            mask=mask,
            sink_size=sink_size,
            recent_size=recent_size,
        )
        eval_cache[key] = result
        return result

    log("Evaluating all-full baseline (all heads set to full attention)...")
    if (not use_distributed) or is_main_process:
        (
            baseline_accuracy,
            baseline_total,
            baseline_exact,
            baseline_detailed_samples,
        ) = eval_with_cache(full_mask)
    else:
        baseline_accuracy = 0.0
        baseline_total = 0
        baseline_exact = 0
        baseline_detailed_samples = []

    if use_distributed:
        baseline_object_list: List[object] = [
            baseline_accuracy,
            baseline_total,
            baseline_exact,
        ]
        dist.broadcast_object_list(baseline_object_list, src=0)
        baseline_accuracy = float(baseline_object_list[0])
        baseline_total = int(baseline_object_list[1])
        baseline_exact = int(baseline_object_list[2])

    baseline_result = (
        baseline_accuracy,
        baseline_total,
        baseline_exact,
        baseline_detailed_samples,
    )
    eval_cache[mask_cache_key(full_mask)] = baseline_result

    log(
        f"Baseline full task accuracy: {baseline_accuracy:.4f} "
        f"({baseline_exact}/{baseline_total})"
    )

    rows: List[Dict[str, object]] = []
    detailed_iterations: List[Dict[str, object]] = []

    def build_iteration_detail(
        row: Dict[str, object],
        detailed_samples: List[Dict[str, Any]],
    ) -> Dict[str, object]:
        detail = dict(row)
        detail["samples"] = detailed_samples
        return detail

    run_specs = [
        ("streaming_first", ordered_streaming_pool),
        ("retrieval_first", ordered_retrieval_pool),
    ]
    expected_runs: Set[str] = {run_name for run_name, _ in run_specs}

    if use_distributed:
        task_specs: List[Tuple[int, str, Sequence[Tuple[int, int]], int]] = []
        for ratio in ratios:
            for run_name, ordered_pool in run_specs:
                task_specs.append((ratio, run_name, ordered_pool, len(ordered_pool)))

        local_tasks = [
            task for idx, task in enumerate(task_specs) if idx % world_size == rank
        ]
        local_task_count = len(local_tasks)
        local_count_tensor = torch.tensor([local_task_count], device=device)
        dist.all_reduce(local_count_tensor, op=dist.ReduceOp.MAX)
        max_steps = int(local_count_tensor.item())
        total_tasks = len(task_specs)
        completed_tasks = 0

        next_ratio_cursor = 0

        def maybe_write_contiguous_ratio_checkpoint() -> None:
            nonlocal next_ratio_cursor
            while next_ratio_cursor < len(ratios):
                ratio = ratios[next_ratio_cursor]
                ratio_runs = {
                    str(row["run_name"])
                    for row in rows
                    if int(row["ratio_percent"]) == ratio
                }
                if not expected_runs.issubset(ratio_runs):
                    break
                next_ratio_cursor += 1
                max_contiguous_ratio = ratios[next_ratio_cursor - 1]
                checkpoint_rows = [
                    row
                    for row in rows
                    if int(row["ratio_percent"]) <= max_contiguous_ratio
                ]
                checkpoint_detailed_iterations = [
                    detail
                    for detail in detailed_iterations
                    if int(detail["ratio_percent"]) <= max_contiguous_ratio
                ]
                sort_rows_for_output(checkpoint_rows)
                sort_detailed_iterations_for_output(checkpoint_detailed_iterations)
                checkpoint_report = build_report(
                    args=args,
                    rows=checkpoint_rows,
                    warnings=warnings,
                    ratios=ratios,
                    baseline_accuracy=baseline_accuracy,
                    baseline_exact=baseline_exact,
                    baseline_total=baseline_total,
                    streaming_pool_size=len(streaming_pool),
                    retrieval_pool_size=len(retrieval_pool),
                    actual_sparsity=float(actual_sparsity),
                    sink_size=sink_size,
                    recent_size=recent_size,
                    num_layers=num_layers,
                    num_heads=num_heads,
                )
                write_outputs(
                    args=args,
                    report=checkpoint_report,
                    rows=checkpoint_rows,
                    baseline_accuracy=baseline_accuracy,
                    detailed_iterations=checkpoint_detailed_iterations,
                    emit_path_logs=False,
                    checkpoint_tag=(
                        f"distributed_progress={completed_tasks}/{total_tasks}, "
                        f"ratio={max_contiguous_ratio}%"
                    ),
                )

        for step in range(max_steps):
            payload: Optional[Dict[str, object]] = None
            if step < local_task_count:
                ratio, run_name, ordered_pool, pool_size = local_tasks[step]
                ablated_mask, num_heads_set_streaming = ablate_pool_ratio(
                    full_mask,
                    ordered_pool,
                    ratio_percent=ratio,
                )

                if num_heads_set_streaming == 0:
                    accuracy, total, exact, detailed_samples = baseline_result
                else:
                    accuracy, total, exact, detailed_samples = eval_with_cache(ablated_mask)

                row = {
                    "run_name": run_name,
                    "ratio_percent": ratio,
                    "pool_size": pool_size,
                    "num_heads_set_streaming": num_heads_set_streaming,
                    "accuracy": accuracy,
                    "delta_vs_full_baseline": accuracy - baseline_accuracy,
                    "exact_match_count": exact,
                    "processed_samples": total,
                }
                payload = {
                    "row": row,
                    "detail_json": json.dumps(
                        build_iteration_detail(row, detailed_samples),
                        ensure_ascii=True,
                    ),
                }
                print(
                    f"[rank={rank}] [{run_name}] ratio={ratio:>3}% "
                    f"num_streaming={num_heads_set_streaming:>3} "
                    f"task_acc={accuracy:.4f} "
                    f"delta={accuracy - baseline_accuracy:+.4f}"
                )

            gathered_payloads: List[Optional[Dict[str, object]]] = [None] * world_size
            dist.all_gather_object(gathered_payloads, payload)

            if is_main_process:
                for item in gathered_payloads:
                    if item is None:
                        continue
                    row = item["row"]  # type: ignore[index]
                    detail_json = str(item["detail_json"])  # type: ignore[index]
                    detail = json.loads(detail_json)
                    if (
                        int(detail["num_heads_set_streaming"]) == 0
                        and not detail.get("samples")
                    ):
                        detail = dict(detail)
                        detail["samples"] = baseline_detailed_samples
                    rows.append(row)
                    detailed_iterations.append(detail)
                    completed_tasks += 1

                maybe_write_contiguous_ratio_checkpoint()
    else:
        for ratio in ratios:
            print(f"\nStarting ratio={ratio}%")
            for run_name, ordered_pool in run_specs:
                pool_size = len(ordered_pool)
                ablated_mask, num_heads_set_streaming = ablate_pool_ratio(
                    full_mask,
                    ordered_pool,
                    ratio_percent=ratio,
                )

                if num_heads_set_streaming == 0:
                    accuracy, total, exact, detailed_samples = baseline_result
                else:
                    accuracy, total, exact, detailed_samples = eval_with_cache(ablated_mask)

                row = {
                    "run_name": run_name,
                    "ratio_percent": ratio,
                    "pool_size": pool_size,
                    "num_heads_set_streaming": num_heads_set_streaming,
                    "accuracy": accuracy,
                    "delta_vs_full_baseline": accuracy - baseline_accuracy,
                    "exact_match_count": exact,
                    "processed_samples": total,
                }
                rows.append(row)
                detailed_iterations.append(build_iteration_detail(row, detailed_samples))
                print(
                    f"[{run_name}] ratio={ratio:>3}% "
                    f"num_streaming={num_heads_set_streaming:>3} "
                    f"task_acc={accuracy:.4f} "
                    f"delta={accuracy - baseline_accuracy:+.4f}"
                )

            checkpoint_rows = list(rows)
            checkpoint_detailed_iterations = list(detailed_iterations)
            sort_rows_for_output(checkpoint_rows)
            sort_detailed_iterations_for_output(checkpoint_detailed_iterations)
            checkpoint_report = build_report(
                args=args,
                rows=checkpoint_rows,
                warnings=warnings,
                ratios=ratios,
                baseline_accuracy=baseline_accuracy,
                baseline_exact=baseline_exact,
                baseline_total=baseline_total,
                streaming_pool_size=len(streaming_pool),
                retrieval_pool_size=len(retrieval_pool),
                actual_sparsity=float(actual_sparsity),
                sink_size=sink_size,
                recent_size=recent_size,
                num_layers=num_layers,
                num_heads=num_heads,
            )
            write_outputs(
                args=args,
                report=checkpoint_report,
                rows=checkpoint_rows,
                baseline_accuracy=baseline_accuracy,
                detailed_iterations=checkpoint_detailed_iterations,
                emit_path_logs=False,
                checkpoint_tag=f"ratio={ratio}%",
            )

    if use_distributed:
        dist.barrier()
        if not is_main_process:
            cleanup_distributed(use_distributed)
            return

    sort_rows_for_output(rows)
    sort_detailed_iterations_for_output(detailed_iterations)
    report = build_report(
        args=args,
        rows=rows,
        warnings=warnings,
        ratios=ratios,
        baseline_accuracy=baseline_accuracy,
        baseline_exact=baseline_exact,
        baseline_total=baseline_total,
        streaming_pool_size=len(streaming_pool),
        retrieval_pool_size=len(retrieval_pool),
        actual_sparsity=float(actual_sparsity),
        sink_size=sink_size,
        recent_size=recent_size,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    diagnostics = report["diagnostics"]
    streaming_max_abs_drop = float(diagnostics["streaming_run_max_abs_drop"])
    retrieval_run_drop_at_100 = float(diagnostics["retrieval_run_drop_at_100"])
    retrieval_minus_streaming_drop_at_100 = float(
        diagnostics["retrieval_minus_streaming_drop_at_100"]
    )

    print("\nSummary Report")
    print(
        f"- Baseline full task accuracy: {baseline_accuracy:.4f} "
        f"({baseline_exact}/{baseline_total})"
    )
    print(f"- Streaming pool size: {len(streaming_pool)}")
    print(f"- Retrieval pool size: {len(retrieval_pool)}")
    print(f"- streaming_run_max_abs_drop: {streaming_max_abs_drop:.4f}")
    print(f"- retrieval_run_drop_at_100: {retrieval_run_drop_at_100:.4f}")
    print(
        "- retrieval_minus_streaming_drop_at_100: "
        f"{retrieval_minus_streaming_drop_at_100:.4f}"
    )
    for warning in warnings:
        print(f"- Warning: {warning}")

    write_outputs(
        args=args,
        report=report,
        rows=rows,
        baseline_accuracy=baseline_accuracy,
        detailed_iterations=detailed_iterations,
        emit_path_logs=True,
    )
    cleanup_distributed(use_distributed)

if __name__ == "__main__":
    main()
