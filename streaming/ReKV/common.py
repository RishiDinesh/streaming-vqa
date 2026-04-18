from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(frozen=True)
class StreamingConversation:
    question: str
    answer: str
    start_time: float
    end_time: float
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamingVideoSample:
    sample_id: str
    video_id: str
    video_path: str
    duration: float
    conversations: list[StreamingConversation]
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: Any) -> str:
    return " ".join(str(text).strip().lower().split())


def normalized_exact_match(prediction: Any, reference: Any) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def tokenize_text(text: Any) -> list[str]:
    return re.findall(r"\w+", normalize_text(text))


def token_overlap_scores(prediction: Any, reference: Any) -> dict[str, float]:
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    if not pred_tokens and not ref_tokens:
        return {
            "token_precision": 1.0,
            "token_recall": 1.0,
            "token_f1": 1.0,
        }
    if not pred_tokens or not ref_tokens:
        return {
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
        }

    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "token_precision": float(precision),
        "token_recall": float(recall),
        "token_f1": float(f1),
    }


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(prev[j - 1] + 1)
            else:
                current.append(max(current[-1], prev[j]))
        prev = current
    return prev[-1]


def rouge_l_scores(prediction: Any, reference: Any) -> dict[str, float]:
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    if not pred_tokens and not ref_tokens:
        return {
            "rouge_l_precision": 1.0,
            "rouge_l_recall": 1.0,
            "rouge_l_f1": 1.0,
        }
    if not pred_tokens or not ref_tokens:
        return {
            "rouge_l_precision": 0.0,
            "rouge_l_recall": 0.0,
            "rouge_l_f1": 0.0,
        }

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "rouge_l_precision": float(precision),
        "rouge_l_recall": float(recall),
        "rouge_l_f1": float(f1),
    }


def contains_reference_score(prediction: Any, reference: Any) -> float:
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    if not pred_norm and not ref_norm:
        return 1.0
    if not pred_norm or not ref_norm:
        return 0.0
    return float(ref_norm in pred_norm or pred_norm in ref_norm)


def open_ended_score_bundle(prediction: Any, reference: Any) -> dict[str, float]:
    scores = {
        "normalized_exact_match": normalized_exact_match(prediction, reference),
        "contains_reference": contains_reference_score(prediction, reference),
    }
    scores.update(token_overlap_scores(prediction, reference))
    scores.update(rouge_l_scores(prediction, reference))
    return scores


def aggregate_score_bundles(score_bundles: list[dict[str, float]]) -> dict[str, float | None]:
    if not score_bundles:
        return {}
    metric_keys = sorted(score_bundles[0].keys())
    aggregated: dict[str, float | None] = {}
    for key in metric_keys:
        values = [bundle.get(key) for bundle in score_bundles if bundle.get(key) is not None]
        aggregated[f"avg_{key}"] = float(sum(values) / len(values)) if values else None
    if aggregated.get("avg_judge_score") is not None:
        aggregated["primary_quality_metric"] = "avg_judge_score"
        aggregated["primary_quality_score"] = aggregated["avg_judge_score"]
    elif aggregated.get("avg_rouge_l_f1") is not None:
        aggregated["primary_quality_metric"] = "avg_rouge_l_f1"
        aggregated["primary_quality_score"] = aggregated["avg_rouge_l_f1"]
    elif aggregated.get("avg_token_f1") is not None:
        aggregated["primary_quality_metric"] = "avg_token_f1"
        aggregated["primary_quality_score"] = aggregated["avg_token_f1"]
    return aggregated
