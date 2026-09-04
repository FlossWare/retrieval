"""Apache Solr retrieval and indexing backends.

Solr is treated as a derived search index. Canonical documents and chunks remain
owned by the storage capability; this module only indexes and retrieves them.
The implementation uses Solr's JSON request API and the Python standard
library so the core retrieval package does not acquire a vendor client
runtime dependency.

The endpoint is trusted configuration. Query strings are bounded but are still
passed to Solr's configured query parser and must be validated by callers when
exposed to untrusted users.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .types import RetrievalResult

_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_MAX_LIMIT = 1000
_DEFAULT_MAX_VECTOR_DIMENSIONS = 4096
_DEFAULT_MAX_QUERY_LENGTH = 4096
_DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_ERROR_BYTES = 4096


class SolrError(RuntimeError):
    """Raised when Solr cannot service a retrieval or indexing request."""


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


def _validate_positive_int(value: int, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _validate_limit(limit: int, maximum: int) -> int:
    return _validate_positive_int(limit, "limit", maximum)


def _validate_field(field: str) -> str:
    if not _FIELD_RE.fullmatch(field):
        raise ValueError("field must be a simple Solr field name")
    return field


class _SolrHTTP:
    """Small async adapter around urllib used by both Solr capabilities."""

    def __init__(self, endpoint: str, collection: str, timeout: float, *, max_response_bytes: int) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        _validate_positive_int(max_response_bytes, "max_response_bytes")
        self.endpoint = _validate_endpoint(endpoint)
        self.collection = _validate_collection(collection)
        self.timeout = float(timeout)
        self.max_response_bytes = max_response_bytes

    def url(self, path: str) -> str:
        return f"{self.endpoint}/{quote(self.collection, safe='')}{path}"

    async def post_json(self, url: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        return await asyncio.to_thread(self._post_json, url, body)

    async def post(self, url: str, body: bytes, content_type: str) -> None:
        await asyncio.to_thread(self._post, url, body, content_type)

    def _read_response(self, response: Any) -> bytes:
        body = response.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise SolrError(f"Solr response exceeds {self.max_response_bytes} bytes")
        return body

    def _request(self, url: str, body: bytes, content_type: str) -> Any:
        request = Request(url, data=body, headers={"Content-Type": content_type}, method="POST")
        try:
            return urlopen(request, timeout=self.timeout)
        except HTTPError as exc:
            try:
                detail = exc.read(_DEFAULT_MAX_ERROR_BYTES).decode("utf-8", errors="replace").strip()
            except OSError:
                detail = ""
            suffix = f": {detail[:_DEFAULT_MAX_ERROR_BYTES]}" if detail else ""
            raise SolrError(f"Solr HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise SolrError(f"Solr request failed: {exc.reason}") from exc

    def _post_json(self, url: str, body: bytes) -> dict[str, Any]:
        with self._request(url, body, "application/json") as response:
            raw = self._read_response(response)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SolrError("Solr returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise SolrError("Solr response must be a JSON object")
        return value

    def _post(self, url: str, body: bytes, content_type: str) -> None:
        with self._request(url, body, content_type) as response:
            self._read_response(response)


class SolrRetriever:
    """Retrieve indexed chunks from an Apache Solr collection."""

    def __init__(
        self,
        endpoint: str,
        collection: str,
        *,
        timeout: float = 10.0,
        max_limit: int = _DEFAULT_MAX_LIMIT,
        max_vector_dimensions: int = _DEFAULT_MAX_VECTOR_DIMENSIONS,
        max_query_length: int = _DEFAULT_MAX_QUERY_LENGTH,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self._http = _SolrHTTP(endpoint, collection, timeout, max_response_bytes=max_response_bytes)
        self.max_limit = _validate_positive_int(max_limit, "max_limit")
        self.max_vector_dimensions = _validate_positive_int(max_vector_dimensions, "max_vector_dimensions")
        self.max_query_length = _validate_positive_int(max_query_length, "max_query_length")

    @property
    def endpoint(self) -> str:
        return self._http.endpoint

    @property
    def collection(self) -> str:
        return self._http.collection

    async def search(self, query: str, *, limit: int = 10) -> list[RetrievalResult]:
        """Run lexical retrieval using Solr's configured query parser.

        The query is still Solr syntax, not an escaped user-search string. Callers
        must validate or constrain untrusted input before crossing this boundary.
        """
        if not query.strip():
            return []
        if len(query) > self.max_query_length:
            raise ValueError(f"query must be <= {self.max_query_length} characters")
        limit = _validate_limit(limit, self.max_limit)
        payload = {
            "query": query,
            "limit": limit,
            "fields": ["chunk_id", "content", "source", "metadata", "score", "id"],
        }
        response = await self._http.post_json(self._http.url("/query"), payload)
        return self._results(response)

    async def search_vector(
        self, vector: Sequence[float], *, limit: int = 10, field: str = "embedding"
    ) -> list[RetrievalResult]:
        """Run Solr k-nearest-neighbor vector retrieval."""
        if not vector:
            return []
        limit = _validate_limit(limit, self.max_limit)
        field = _validate_field(field)
        if len(vector) > self.max_vector_dimensions:
            raise ValueError(f"vector must have <= {self.max_vector_dimensions} dimensions")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError("vector values must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vector values must be finite numbers")
        vector_text = ",".join(str(value) for value in values)
        payload = {
            "query": f"{{!knn f={field} topK={limit}}}[{vector_text}]",
            "limit": limit,
            "fields": ["chunk_id", "content", "source", "metadata", "score", "id"],
        }
        response = await self._http.post_json(self._http.url("/query"), payload)
        return self._results(response)

    def _results(self, response: Mapping[str, Any]) -> list[RetrievalResult]:
        response_body = response.get("response", {})
        if not isinstance(response_body, Mapping):
            raise SolrError("Solr response field must be an object")
        documents = response_body.get("docs", [])
        if not isinstance(documents, list):
            raise SolrError("Solr response docs field must be a list")
        return [self._result(doc) for doc in documents if isinstance(doc, Mapping)]

    @staticmethod
    def _result(doc: Mapping[str, Any]) -> RetrievalResult:
        known = {"chunk_id", "content", "source", "metadata", "score", "id"}
        raw_metadata = doc.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata.update({key: value for key, value in doc.items() if key not in known})
        raw_score = doc.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise SolrError("Solr result score must be numeric") from exc
        chunk_id = doc.get("chunk_id", doc.get("id", ""))
        return RetrievalResult(
            content=str(doc.get("content", "")),
            score=score,
            source=str(doc.get("source", "")),
            chunk_id=str(chunk_id),
            metadata=metadata,
        )


class SolrIndexer:
    """Index and delete derived chunk documents in Apache Solr.

    The Solr unique-key field is `id` and must contain the canonical chunk ID.
    If a document supplies `chunk_id` instead, it is copied into `id`; if both
    are supplied they must agree. This keeps deletion and retrieval identity
    aligned with canonical storage.
    """

    def __init__(
        self,
        endpoint: str,
        collection: str,
        *,
        timeout: float = 10.0,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self._http = _SolrHTTP(endpoint, collection, timeout, max_response_bytes=max_response_bytes)

    async def index(self, documents: Sequence[Mapping[str, Any]]) -> None:
        """Index documents/chunks using Solr's JSON update endpoint."""
        if not documents:
            return
        normalized: list[dict[str, Any]] = []
        for document in documents:
            item = dict(document)
            chunk_id = item.get("chunk_id")
            document_id = item.get("id")
            if chunk_id is not None and not isinstance(chunk_id, str):
                raise ValueError("chunk_id must be a non-empty string")
            if document_id is not None and not isinstance(document_id, str):
                raise ValueError("id must be a non-empty string")
            if chunk_id is None and document_id is None:
                raise ValueError("indexed documents must contain canonical chunk_id or id")
            if chunk_id is not None and not chunk_id:
                raise ValueError("chunk_id must be a non-empty string")
            if document_id is not None and not document_id:
                raise ValueError("id must be a non-empty string")
            if chunk_id is not None and document_id is not None and chunk_id != document_id:
                raise ValueError("id and chunk_id must contain the same canonical chunk ID")
            item["id"] = chunk_id if chunk_id is not None else document_id
            normalized.append(item)
        url = self._http.url("/update/json/docs?commit=true")
        body = json.dumps(normalized).encode("utf-8")
        await self._http.post(url, body, "application/json")

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete indexed chunks by their canonical chunk IDs."""
        if not chunk_ids:
            return
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids):
            raise ValueError("chunk_ids must contain non-empty strings")
        url = self._http.url("/update?commit=true")
        body = json.dumps({"delete": list(chunk_ids)}).encode("utf-8")
        await self._http.post(url, body, "application/json")
