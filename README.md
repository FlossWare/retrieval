# retrieval

Reusable retrieval capability for FlossWare.

This repository owns retrieval behavior and contracts, including lexical, vector, and hybrid ranking. Storage and embedding/model providers are dependencies behind interfaces, not implementation details of the calling application.

## Architectural boundary

Retrieval answers **which supplied records are relevant**. It does not own document ingestion, persistence implementation, durable Knowledge, model-provider routing, or Loom orchestration.

```text
documents/chunks -> retrieval -> ranked evidence
                       ^
                       |
                storage/index adapters
```

- `chunking` owns canonical chunk creation.
- `storage` owns persistence contracts and adapters.
- `knowledge` owns durable, versioned knowledge.
- `rag` composes retrieval into retrieval-augmented context pipelines.
- `loom-ai` owns Workers, Arbiters, execution, and orchestration.

The initial reference implementation remains dependency-free and operates on supplied records, making it suitable for tests and lightweight deployments.
