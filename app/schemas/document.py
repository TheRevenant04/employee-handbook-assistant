from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Document:
    path: str
    content: str
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    document_path: str
    content: str
    chunk_index: int
    embedding: np.ndarray | None = None
