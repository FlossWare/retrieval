import asyncio
import json

import pytest

from retrieval import SolrIndexer, SolrRetriever
import retrieval.solr as solr


class FakeResponse:
    def __init__(self, payload=None, raw=None):
        self.payload = payload
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        if self.raw is not None:
            return self.raw if size < 0 else self.raw[:size]
        return json.dumps(self.payload).encode("utf-8") if self.payload is not None else b""


def test_solr_search_builds_json_request(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse({"response": {"docs": [{"id": "c1", "content": "hello", "source": "d1", "score": 2.5}]}})

    monkeypatch.setattr(solr, "urlopen", fake_urlopen)

    async def run():
        return await SolrRetriever("http://localhost:8983/solr", "chunks", timeout=3).search("hello", limit=4)

    results = asyncio.run(run())
    assert results[0].chunk_id == "c1"
    assert results[0].score == 2.5
    assert requests[0][1] == 3.0
    assert requests[0][0].full_url == "http://localhost:8983/solr/chunks/query"
    assert json.loads(requests[0][0].data)["limit"] == 4


def test_solr_vector_search_validates_field_values_and_dimension(monkeypatch):
    monkeypatch.setattr(solr, "urlopen", lambda request, timeout: FakeResponse({"response": {"docs": []}}))

    async def run():
        retriever = SolrRetriever("http://localhost:8983/solr", "chunks", max_vector_dimensions=2)
        with pytest.raises(ValueError):
            await retriever.search_vector([1.0], field="embedding;delete")
        with pytest.raises(ValueError):
            await retriever.search_vector([float("nan")])
        with pytest.raises(ValueError):
            await retriever.search_vector([1.0, 2.0, 3.0])

    asyncio.run(run())


def test_solr_limit_and_query_bounds(monkeypatch):
    monkeypatch.setattr(solr, "urlopen", lambda request, timeout: FakeResponse({"response": {"docs": []}}))

    async def run():
        retriever = SolrRetriever("http://localhost:8983/solr", "chunks", max_limit=2, max_query_length=3)
        with pytest.raises(ValueError):
            await retriever.search("hello")
        with pytest.raises(ValueError):
            await retriever.search("x", limit=3)

    asyncio.run(run())


def test_solr_validation():
    with pytest.raises(ValueError):
        SolrRetriever("ftp://localhost:8983/solr", "chunks")
    with pytest.raises(ValueError):
        SolrRetriever("http://user:pass@localhost:8983/solr", "chunks")
    with pytest.raises(ValueError):
        SolrRetriever("http://localhost:8983/solr", "a/b")
    with pytest.raises(ValueError):
        SolrRetriever("http://localhost:8983/solr", "chunks", timeout=0)


def test_solr_indexer_uses_valid_delete_payload(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(solr, "urlopen", fake_urlopen)

    async def run():
        indexer = SolrIndexer("http://localhost:8983/solr", "chunks")
        await indexer.index([{"id": "c1", "content": "hello"}])
        await indexer.delete(["c1", "c2"])

    asyncio.run(run())
    assert requests[0][0].full_url.endswith("/chunks/update/json/docs?commit=true")
    assert requests[1][0].full_url.endswith("/chunks/update?commit=true")
    assert json.loads(requests[1][0].data) == {"delete": ["c1", "c2"]}


def test_solr_indexer_normalizes_canonical_chunk_identity(monkeypatch):
    requests = []
    monkeypatch.setattr(solr, "urlopen", lambda request, timeout: (requests.append(request) or FakeResponse()))

    async def run():
        indexer = SolrIndexer("http://localhost:8983/solr", "chunks")
        await indexer.index([{"chunk_id": "c1", "content": "hello"}])
        with pytest.raises(ValueError):
            await indexer.index([{"id": "c1", "chunk_id": "c2"}])
        with pytest.raises(ValueError):
            await indexer.index([{"content": "missing identity"}])

    asyncio.run(run())
    assert json.loads(requests[0].data)[0]["id"] == "c1"


def test_solr_response_size_is_bounded(monkeypatch):
    monkeypatch.setattr(solr, "urlopen", lambda request, timeout: FakeResponse(raw=b"x" * 11))

    async def run():
        retriever = SolrRetriever("http://localhost:8983/solr", "chunks", max_response_bytes=10)
        with pytest.raises(solr.SolrError, match="exceeds 10 bytes"):
            await retriever.search("hello")

    asyncio.run(run())
