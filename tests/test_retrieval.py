import asyncio

from retrieval import InMemoryRetriever


def test_keyword_retrieval():
    async def run():
        r = InMemoryRetriever()
        r.add_chunk("c1", "PostgreSQL uses pgvector", source="d1")
        r.add_chunk("c2", "Redis is a cache", source="d2")
        results = await r.search("pgvector", mode="keyword")
        assert results[0].chunk_id == "c1"

    asyncio.run(run())


def test_vector_retrieval():
    async def run():
        r = InMemoryRetriever()
        r.add_chunk("c1", "one", source="d1")
        r.add_chunk("c2", "two", source="d2")
        r.add_vector("c1", [1.0, 0.0])
        r.add_vector("c2", [0.0, 1.0])
        results = await r.search("ignored", mode="vector", vector=[1.0, 0.0])
        assert results[0].chunk_id == "c1"
        assert results[0].score == 1.0

    asyncio.run(run())
