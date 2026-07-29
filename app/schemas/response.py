from dataclasses import dataclass
from typing import Any


@dataclass
class Answer:
    text: str
    message_id: int | None = None
    conversation_id: int | None = None
    sources: list[dict[str, Any]] = None


@dataclass
class EvaluationScores:
    faithfulness_score: int
    faithfulness_reasoning: str
    context_relevance_score: int
    context_relevance_reasoning: str
    completeness_score: int
    completeness_reasoning: str
