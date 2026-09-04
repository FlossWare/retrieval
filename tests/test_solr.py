import asyncio
import json

import pytest

from retrieval import SolrIndexer, SolrRetriever
import retrieval.solr as solr


class FakeResponse:
    def __init__(self, payload=None):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
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


def test_solr_vector_search_validates_field_and_values(monkeypatch):
    monkeypatch.setattr(solr, "urlopen", lambda request, timeout: FakeResponse({"response": {"docs": []}}))

    async def run():
        retriever = SolrRetriever("http://localhost:8983/solr", "chunks")
        with pytest.raises(ValueError):
            await retriever.search_vector([1.0], field="embedding;delete")
        with pytest.raises(ValueError):
            await retriever.search_vector([float("nan")])

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


def test_solr_indexer_uses_update_endpoint(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(solr, "urlopen", fake_urlopen)

    async def run():
        indexer = SolrIndexer("http://localhost:8983/solr", "chunks")
        await indexer.index([{"id": "c1", "content": "hello"}])
        await indexer.delete(["c1"])

    asyncio.run(run())
    assert requests[0][0].full_url.endswith("/chunks/update/json/docs?commit=true")
    assert requests[1][0].full_url.endswith("/chunks/update?commit=true")
    assert json.loads(requests[1][0].data) == {"delete": [{"id": "c1"}]}
