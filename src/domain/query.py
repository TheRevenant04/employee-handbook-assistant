from dataclasses import dataclass, field
from typing import Any


@dataclass
class Query:
    text: str
    rewritten_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    id: int
    path: str
    content: str
    distance: float = 0.0
    score: float = 0.0
    score_type: str = "distance"
    method: str = "vector"
    rerank_score: float | None = None


