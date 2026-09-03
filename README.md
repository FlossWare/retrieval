# retrieval

Retrieval capability for FlossWare.

This repository owns retrieval behavior and contracts, including lexical, vector, and hybrid ranking. Storage and model providers are dependencies behind interfaces, not implementation details of the calling application.

The initial reference implementation is dependency-free and operates on supplied records, making it suitable for tests and lightweight deployments.
