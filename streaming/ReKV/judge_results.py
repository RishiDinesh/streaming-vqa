#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .common import aggregate_score_bundles
from .methods import (
    ensure_generation_pad_token,
    get_stop_token_ids,
    greedy_decode_with_cache,
    resolve_device,
    resolve_dtype,
)


DEFAULT_JUDGE_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
PROMPT_VERSION = "judge_v2_score_only_chat"

SCORE_LABELS = {
    5: "fully_correct",
    4: "mostly_correct",
    3: "partially_correct",
    2: "weakly_related",
    1: "mostly_incorrect",
    0: "wrong_or_empty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc local LLM judge rescoring for streaming VQA result JSON files."
    )
    parser.add_argument("result_paths", nargs="+", help="One or more *_results.json files.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    return parser.parse_args()


def build_judge_prompt(tokenizer, question: str, reference_answer: str, prediction: str) -> str:
    system_prompt = (
        "You are a strict evaluator for streaming video question answering. "
        "Score only semantic correctness against the reference answer."
    )
    user_prompt = (
        "Score the model answer from 0 to 5 using this rubric:\n"
        "5 = fully correct\n"
        "4 = mostly correct\n"
        "3 = partially correct\n"
        "2 = weakly related\n"
        "1 = mostly incorrect\n"
        "0 = wrong, contradictory, or empty\n\n"
        "Respond with a single digit only.\n\n"
        f"QUESTION: {question}\n"
        f"REFERENCE: {reference_answer}\n"
        f"MODEL_ANSWER: {prediction}\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return prompt + "SCORE: "


def parse_judge_output(output_text: str) -> tuple[float | None, dict[str, object]]:
    score_match = re.search(r"SCORE:\s*([0-5])", output_text, flags=re.IGNORECASE)
    if score_match is None:
        fallback = re.search(r"\b([0-5])\b", output_text)
        if fallback is not None:
            score_match = fallback

    if score_match is None:
        return None, {
            "parse_success": False,
            "raw_output": output_text.strip(),
            "label": "",
            "reason": "",
        }

    raw_score = int(score_match.group(1))
    return raw_score / 5.0, {
        "parse_success": True,
        "raw_output": output_text.strip(),
        "score_raw": raw_score,
        "label": SCORE_LABELS[raw_score],
        "reason": "",
    }


class LocalLlmJudge:
    def __init__(
        self,
        *,
        judge_model: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
    ) -> None:
        self.model_name = judge_model
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.max_new_tokens = int(max_new_tokens)
        self.processor = AutoProcessor.from_pretrained(judge_model, trust_remote_code=True)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            judge_model,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()
        ensure_generation_pad_token(self.model, self.processor)
        self.tokenizer = self.processor.tokenizer
        self.stop_token_ids = get_stop_token_ids(self.model, self.tokenizer)

    def score(self, *, question: str, reference_answer: str, prediction: str) -> dict[str, object]:
        prompt = build_judge_prompt(self.tokenizer, question, reference_answer, prediction)
        output_text, decode_stats = greedy_decode_with_cache(
            language_model=self.model.language_model,
            output_projection=self.model.get_output_embeddings(),
            tokenizer=self.tokenizer,
            prompt_text=prompt,
            past_key_values=None,
            stop_token_ids=self.stop_token_ids,
            max_new_tokens=self.max_new_tokens,
            device=self.device,
        )
        normalized_score, parsed = parse_judge_output(output_text)
        return {
            "judge_model": self.model_name,
            "prompt_version": PROMPT_VERSION,
            "judge_score": normalized_score,
            "decode_stats": decode_stats,
            **parsed,
        }


def judge_payload(payload: dict, judge: LocalLlmJudge, *, show_progress_bar: bool) -> dict:
    score_bundles: list[dict[str, float]] = []
    judge_items = [
        (video, conversation)
        for video in payload.get("videos", [])
        for conversation in video.get("conversations", [])
    ]
    iterator = judge_items
    if show_progress_bar and tqdm is not None:
        iterator = tqdm(judge_items, desc="judge conversations", unit="conv")

    for video, conversation in iterator:
        scores = dict(conversation.get("scores", {}))
        judge_result = judge.score(
            question=conversation.get("question", ""),
            reference_answer=conversation.get("reference_answer", ""),
            prediction=conversation.get("prediction", ""),
        )
        judge_score = judge_result.get("judge_score")
        if judge_score is not None:
            scores["judge_score"] = float(judge_score)
        scores["judge_parse_success"] = float(bool(judge_result.get("parse_success")))
        conversation["scores"] = scores
        conversation["judge"] = judge_result
        score_bundles.append(scores)

    aggregate_metrics = dict(payload.get("aggregate_metrics", {}))
    aggregate_metrics.update(aggregate_score_bundles(score_bundles))
    if aggregate_metrics.get("avg_normalized_exact_match") is not None:
        aggregate_metrics["normalized_exact_match"] = aggregate_metrics["avg_normalized_exact_match"]
    aggregate_metrics["evaluation_mode"] = "open_ended_bundle_plus_judge"
    payload["aggregate_metrics"] = aggregate_metrics
    payload["judge_config"] = {
        "judge_model": judge.model_name,
        "prompt_version": PROMPT_VERSION,
        "max_new_tokens": judge.max_new_tokens,
    }
    return payload


def main() -> int:
    args = parse_args()
    judge = LocalLlmJudge(
        judge_model=args.judge_model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    for raw_path in args.result_paths:
        path = Path(raw_path).expanduser().resolve()
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        judged = judge_payload(payload, judge, show_progress_bar=not args.disable_progress_bar)
        if args.in_place:
            output_path = path
        else:
            output_path = path.with_name(f"{path.stem}_judged{path.suffix}")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(judged, handle, indent=2)
        print(f"Saved judged results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
