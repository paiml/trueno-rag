<div align="center">

<img src=".github/trueno-rag-hero.svg" alt="trueno-rag" width="600">

**Pure-Rust Retrieval-Augmented Generation Pipeline**

[![CI](https://github.com/paiml/trueno-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/trueno-rag/actions)

</div>

---

SIMD-accelerated RAG pipeline built on Trueno compute primitives.

## Features

- **Pure Rust** - Zero Python/C++ dependencies
- **Chunking** - Recursive, Fixed, Sentence, Paragraph, Semantic, Structural
- **Hybrid Retrieval** - Dense (vector) + Sparse (BM25) search
- **Fusion** - RRF, Linear, DBSF, Convex, Union, Intersection
- **Reranking** - Lexical, cross-encoder, and composite rerankers
- **Metrics** - Recall, Precision, MRR, NDCG, MAP

## Installation

```toml
[dependencies]
trueno-rag = "0.1"
```

## Quick Start

```rust
use trueno_rag::{
    pipeline::RagPipelineBuilder,
    chunk::RecursiveChunker,
    embed::MockEmbedder,
    rerank::LexicalReranker,
    fusion::FusionStrategy,
    Document,
};

let mut pipeline = RagPipelineBuilder::new()
    .chunker(RecursiveChunker::new(512, 50))
    .embedder(MockEmbedder::new(384))
    .reranker(LexicalReranker::new())
    .fusion(FusionStrategy::RRF { k: 60.0 })
    .build()?;

let doc = Document::new("Your content here...").with_title("Doc Title");
pipeline.index_document(&doc)?;

let (results, context) = pipeline.query_with_context("your query", 5)?;
```

## Examples

```bash
cargo run --example basic_rag
cargo run --example chunking_strategies
cargo run --example hybrid_search
cargo run --example metrics_evaluation
```

## Development

```bash
make test      # Run tests
make lint      # Lint
make coverage  # Coverage report
```

## License

MIT
