"""Apache Solr retrieval and indexing backends.

Solr is treated as a derived search index. Canonical documents and chunks remain
owned by the storage capability; this module only indexes and retrieves them.
The implementation uses Solr's JSON request API and the Python standard
library so the core retrieval package does not acquire a vendor client
runtime dependency.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .types import RetrievalResult

_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint must not contain credentials")
    return value


def _validate_collection(collection: str) -> str:
    value = collection.strip("/")
    if not value or "/" in value or value in {".", ".."}:
        raise ValueError("collection must be a single non-empty path segment")
    return value


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    return limit


def _validate_field(field: str) -> str:
    if not _FIELD_RE.fullmatch(field):
        raise ValueError("field must be a simple Solr field name")
    return field


class _SolrHTTP:
    """Small async adapter around urllib used by both Solr capabilities."""

    def __init__(self, endpoint: str, collection: str, timeout: float) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        self.endpoint = _validate_endpoint(endpoint)
        self.collection = _validate_collection(collection)
        self.timeout = float(timeout)

    def url(self, path: str) -> str:
        return f"{self.endpoint}/{quote(self.collection, safe='')}{path}"

    async def post_json(self, url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        return await asyncio.to_thread(self._post_json, url, body)

    async def post(self, url: str, body: bytes, content_type: str) -> None:
        await asyncio.to_thread(self._post, url, body, content_type)

    def _post_json(self, url: str, body: bytes) -> dict[str, Any]:
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Solr response must be a JSON object")
        return value

    def _post(self, url: str, body: bytes, content_type: str) -> None:
        request = Request(url, data=body, headers={"Content-Type": content_type}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            response.read()


class SolrRetriever:
    """Retrieve indexed chunks from an Apache Solr collection."""

    def __init__(self, endpoint: str, collection: str, *, timeout: float = 10.0) -> None:
        self._http = _SolrHTTP(endpoint, collection, timeout)

    @property
    def endpoint(self) -> str:
        return self._http.endpoint

    @property
    def collection(self) -> str:
        return self._http.collection

    async def search(self, query: str, *, limit: int = 10) -> list[RetrievalResult]:
        """Run lexical retrieval using Solr's configured query parser."""
        if not query.strip():
            return []
        limit = _validate_limit(limit)
        payload = {
            "query": query,
            "limit": limit,
            "fields": ["chunk_id", "content", "source", "metadata", "score"],
        }
        response = await self._http.post_json(self._http.url("/query"), payload)
        return self._results(response)

    async def search_vector(
        self, vector: Sequence[float], *, limit: int = 10, field: str = "embedding"
    ) -> list[RetrievalResult]:
        """Run Solr k-nearest-neighbor vector retrieval."""
        if not vector:
            return []
        limit = _validate_limit(limit)
        field = _validate_field(field)
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vector values must be finite numbers")
        vector_text = ",".join(str(value) for value in values)
        payload = {
            "query": f"{{!knn f={field} topK={limit}}}[{vector_text}]",
            "limit": limit,
            "fields": ["chunk_id", "content", "source", "metadata", "score"],
        }
        response = await self._http.post_json(self._http.url("/query"), payload)
        return self._results(response)

    def _results(self, response: Mapping[str, Any]) -> list[RetrievalResult]:
        response_body = response.get("response", {})
        if not isinstance(response_body, Mapping):
            raise ValueError("Solr response field must be an object")
        documents = response_body.get("docs", [])
        if not isinstance(documents, list):
            raise ValueError("Solr response docs field must be a list")
        return [self._result(doc) for doc in documents if isinstance(doc, Mapping)]

    @staticmethod
    def _result(doc: Mapping[str, Any]) -> RetrievalResult:
        known = {"chunk_id", "content", "source", "metadata", "score"}
        raw_metadata = doc.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata.update({key: value for key, value in doc.items() if key not in known})
        raw_score = doc.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("Solr result score must be numeric") from exc
        return RetrievalResult(
            content=str(doc.get("content", "")),
            score=score,
            source=str(doc.get("source", "")),
            chunk_id=str(doc.get("chunk_id", doc.get("id", ""))),
            metadata=metadata,
        )


class SolrIndexer:
    """Index and delete derived chunk documents in Apache Solr."""

    def __init__(self, endpoint: str, collection: str, *, timeout: float = 10.0) -> None:
        self._http = _SolrHTTP(endpoint, collection, timeout)

    async def index(self, documents: Sequence[Mapping[str, Any]]) -> None:
        """Index documents/chunks using Solr's JSON update endpoint."""
        if not documents:
            return
        url = self._http.url("/update/json/docs?commit=true")
        body = json.dumps(list(documents)).encode("utf-8")
        await self._http.post(url, body, "application/json")

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete indexed chunks by their canonical chunk IDs."""
        if not chunk_ids:
            return
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids):
            raise ValueError("chunk_ids must contain non-empty strings")
        url = self._http.url("/update?commit=true")
        body = json.dumps({"delete": [{"id": chunk_id} for chunk_id in chunk_ids]}).encode("utf-8")
        await self._http.post(url, body, "application/json")
