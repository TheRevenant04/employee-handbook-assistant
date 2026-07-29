from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    @abstractmethod
    def connect(self):
        ...

    @abstractmethod
    def init_schema(self, table_name: str, dim: int):
        ...

    @abstractmethod
    def insert(self, table_name: str, rows: list[tuple]):
        ...

    @abstractmethod
    def vector_search(self, query_vector, table_name: str, num_results: int) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def keyword_search(self, query_text: str, table_name: str, num_results: int) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def hybrid_search(self, query_text: str, query_vector, table_name: str, num_results: int, alpha: float) -> list[dict[str, Any]]:
        ...
