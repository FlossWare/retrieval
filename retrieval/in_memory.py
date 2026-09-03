"""Dependency-free lexical, vector, and hybrid retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .types import RetrievalResult


class InMemoryRetriever:
    """Retrieve supplied chunks using lexical, vector, or hybrid ranking."""

    def __init__(self, *, keyword_weight: float = 0.5, vector_weight: float = 0.5, rrf_k: int = 60) -> None:
        if keyword_weight < 0 or vector_weight < 0:
            raise ValueError("weights must be non-negative")
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self._chunks: dict[str, tuple[str, str, dict[str, Any]]] = {}
        self._vectors: dict[str, list[float]] = {}

    def add_chunk(self, chunk_id: str, content: str, *, source: str = "", metadata: dict[str, Any] | None = None) -> None:
        self._chunks[chunk_id] = (content, source, dict(metadata or {}))

    def add_vector(self, chunk_id: str, vector: Sequence[float]) -> None:
        if chunk_id not in self._chunks:
            raise KeyError(f"unknown chunk: {chunk_id}")
        self._vectors[chunk_id] = list(vector)

    async def search(self, query: str, *, limit: int = 10, mode: str = "hybrid", vector: Sequence[float] | None = None) -> list[RetrievalResult]:
        if mode == "keyword":
            return self._keyword(query, limit)
        if mode == "vector":
            return self._vector(vector if vector is not None else _text_vector(query), limit)
        if mode != "hybrid":
            raise ValueError(f"unsupported retrieval mode: {mode}")
        kw = self._keyword(query, max(limit * 2, limit))
        vec = self._vector(vector if vector is not None else _text_vector(query), max(limit * 2, limit))
        return self._rrf(kw, vec, limit)

    def _keyword(self, query: str, limit: int) -> list[RetrievalResult]:
        words = query.lower().split()
        if not words:
            return []
        scored = []
        for chunk_id, (content, source, metadata) in self._chunks.items():
            score = float(sum(content.lower().count(word) for word in words))
            if score > 0:
                scored.append((chunk_id, score, content, source, metadata))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [RetrievalResult(c, s, src, cid, dict(meta)) for cid, s, c, src, meta in scored[:limit]]

    def _vector(self, vector: Sequence[float], limit: int) -> list[RetrievalResult]:
        scored = []
        for chunk_id, stored in self._vectors.items():
            if chunk_id not in self._chunks:
                continue
            content, source, metadata = self._chunks[chunk_id]
            scored.append((chunk_id, _cosine(vector, stored), content, source, metadata))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [RetrievalResult(c, s, src, cid, dict(meta)) for cid, s, c, src, meta in scored[:limit]]

    def _rrf(self, lexical: list[RetrievalResult], vector: list[RetrievalResult], limit: int) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        results: dict[str, RetrievalResult] = {}
        for rank, result in enumerate(lexical, 1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + self.keyword_weight / (self.rrf_k + rank)
            results[result.chunk_id] = result
        for rank, result in enumerate(vector, 1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + self.vector_weight / (self.rrf_k + rank)
            results.setdefault(result.chunk_id, result)
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
        return [RetrievalResult(results[cid].content, scores[cid], results[cid].source, cid, dict(results[cid].metadata)) for cid in ranked[:limit]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _text_vector(text: str, dim: int = 64) -> list[float]:
    vector = [0.0] * dim
    lower = text.lower()
    for i in range(len(lower) - 1):
        pair = lower[i : i + 2]
        bucket = sum(ord(c) for c in pair) % dim
        vector[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]
