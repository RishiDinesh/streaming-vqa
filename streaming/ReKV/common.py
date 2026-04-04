from __future__ import annotations

from dataclasses import dataclass, field
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
