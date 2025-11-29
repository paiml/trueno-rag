# Trueno-RAG

Pure-Rust Retrieval-Augmented Generation (RAG) pipeline built on Trueno compute primitives.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-261%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## Features

- **Pure Rust** - Zero Python/C++ dependencies
- **SIMD-Accelerated** - Built on Trueno compute primitives
- **Flexible Chunking** - 6 strategies: Recursive, Fixed, Sentence, Paragraph, Semantic, Structural
- **Hybrid Retrieval** - Dense (vector) + Sparse (BM25) search
- **Multiple Fusion** - RRF, Linear, DBSF, Convex, Union, Intersection
- **Reranking** - Lexical, cross-encoder, and composite rerankers
- **Evaluation Metrics** - Recall, Precision, MRR, NDCG, MAP
- **Query Preprocessing** - HyDE, multi-query expansion, synonyms

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

// Build pipeline
let mut pipeline = RagPipelineBuilder::new()
    .chunker(RecursiveChunker::new(512, 50))
    .embedder(MockEmbedder::new(384))
    .reranker(LexicalReranker::new())
    .fusion(FusionStrategy::RRF { k: 60.0 })
    .build()?;

// Index documents
let doc = Document::new("Your content here...").with_title("Doc Title");
pipeline.index_document(&doc)?;

// Query with context
let (results, context) = pipeline.query_with_context("your query", 5)?;
println!("{}", context.format_with_citations());
```

## Installation

```toml
[dependencies]
trueno-rag = "0.1"
```

## Examples

```bash
# Basic RAG pipeline
cargo run --example basic_rag

# Compare chunking strategies
cargo run --example chunking_strategies

# Hybrid search with different fusion
cargo run --example hybrid_search

# Retrieval metrics evaluation
cargo run --example metrics_evaluation
```

## Documentation

```bash
# Build and serve documentation book
make book-serve

# Or build only
make book
```

## Chunking Strategies

| Strategy | Use Case |
|----------|----------|
| `RecursiveChunker` | General purpose (default) |
| `FixedSizeChunker` | Uniform chunks |
| `SentenceChunker` | Preserve sentences |
| `ParagraphChunker` | Paragraph-level retrieval |
| `SemanticChunker` | Topic-based grouping |
| `StructuralChunker` | Markdown/structured docs |

## Fusion Strategies

| Strategy | Description |
|----------|-------------|
| `RRF` | Reciprocal Rank Fusion (default, most robust) |
| `Linear` | Weighted combination |
| `DBSF` | Distribution-based score fusion |
| `Convex` | Convex combination |
| `Union` | Maximum recall |
| `Intersection` | Maximum precision |

## Development

```bash
# Run tests
make test

# Fast tests (release mode)
make test-fast

# Lint
make lint

# Format
make fmt

# Full CI check
make ci

# Coverage report
make coverage
```

## Architecture

```
Document -> Chunker -> Embedder -> Index (Dense + Sparse)
                                        |
Query -> Preprocessor -> Retriever -> Fusion -> Reranker -> Context
```

## License

MIT
