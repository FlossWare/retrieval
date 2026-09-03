"""Retrieval contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class RetrievalResult:
    content: str
    score: float
    source: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    async def search(self, query: str, *, limit: int = 10) -> list[RetrievalResult]: ...


class VectorRetriever(Protocol):
    async def search_vector(self, vector: Sequence[float], *, limit: int = 10) -> list[RetrievalResult]: ...
