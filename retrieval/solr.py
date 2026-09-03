"""Apache Solr retrieval backend.

Solr is treated as a derived search index. Canonical documents and chunks remain
owned by the storage capability; this module only indexes and retrieves them.
The implementation uses Solr's JSON request API and the Python standard
library so the core retrieval package does not acquire a vendor client
runtime dependency.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen

from .types import RetrievalResult


class SolrRetriever:
    """Retrieve indexed chunks from an Apache Solr collection.

    Expected fields are ``chunk_id``, ``content``, ``score`` (returned by Solr),
    ``source`` and optional ``metadata``. Additional fields are preserved in
    metadata where practical.
    """

    def __init__(self, endpoint: str, collection: str, *, timeout: float = 10.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.collection = collection.strip("/")
        self.timeout = timeout

    @property
    def _url(self) -> str:
        return f"{self.endpoint}/{quote(self.collection, safe='')}/query"

    async def search(self, query: str, *, limit: int = 10) -> list[RetrievalResult]:
        """Run lexical/BM25 retrieval using Solr's configured query parser."""
        if not query.strip() or limit <= 0:
            return []
        payload = {
            "query": query,
            "limit": limit,
            "fields": ["chunk_id", "content", "source", "metadata", "score"],
        }
        response = await self._request(payload)
        return [self._result(doc) for doc in response.get("response", {}).get("docs", [])]

    async def search_vector(
        self, vector: Sequence[float], *, limit: int = 10, field: str = "embedding"
    ) -> list[RetrievalResult]:
        """Run Solr k-nearest-neighbor vector retrieval."""
        if not vector or limit <= 0:
            return []
        vector_text = ",".join(str(float(value)) for value in vector)
        payload = {
            "query": f"{{!knn f={field} topK={limit}}}[{vector_text}]",
            "limit": limit,
            "fields": ["chunk_id", "content", "source", "metadata", "score"],
        }
        response = await self._request(payload)
        return [self._result(doc) for doc in response.get("response", {}).get("docs", [])]

    async def index(self, documents: Sequence[Mapping[str, Any]]) -> None:
        """Index documents/chunks into Solr using the JSON update endpoint."""
        if not documents:
            return
        url = f"{self.endpoint}/{quote(self.collection, safe='')}/update/json/docs?commit=true"
        body = json.dumps(list(documents)).encode("utf-8")
        await asyncio.to_thread(self._http, url, body, "application/json")

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete indexed chunks by their canonical chunk IDs."""
        if not chunk_ids:
            return
        url = f"{self.endpoint}/{quote(self.collection, safe='')}/update?commit=true"
        body = json.dumps({"delete": [{"id": chunk_id} for chunk_id in chunk_ids]}).encode("utf-8")
        await asyncio.to_thread(self._http, url, body, "application/json")

    async def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        return await asyncio.to_thread(self._http_json, self._url, body)

    def _result(self, doc: Mapping[str, Any]) -> RetrievalResult:
        known = {"chunk_id", "content", "source", "metadata", "score"}
        metadata = dict(doc.get("metadata") or {})
        metadata.update({key: value for key, value in doc.items() if key not in known})
        return RetrievalResult(
            content=str(doc.get("content", "")),
            score=float(doc.get("score", 0.0)),
            source=str(doc.get("source", "")),
            chunk_id=str(doc.get("chunk_id", doc.get("id", ""))),
            metadata=metadata,
        )

    def _http_json(self, url: str, body: bytes) -> dict[str, Any]:
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _http(self, url: str, body: bytes, content_type: str) -> None:
        request = Request(url, data=body, headers={"Content-Type": content_type}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            response.read()
