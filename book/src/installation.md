# Installation

## Requirements

- Rust 1.70+ (stable)
- Cargo

## From crates.io

```bash
cargo add trueno-rag
```

## From Source

```bash
git clone https://github.com/noahgift/trueno-rag
cd trueno-rag
cargo build --release
```

## Running Tests

```bash
# All tests
cargo test

# Fast tests (release mode)
make test-fast

# With coverage
make coverage
```

## Running Examples

Trueno-RAG includes several examples demonstrating key features:

```bash
# Run all examples
make examples

# Run individual examples
cargo run --example basic_rag
cargo run --example chunking_strategies
cargo run --example hybrid_search
cargo run --example metrics_evaluation
```

### Available Examples

| Example | Description |
|---------|-------------|
| `basic_rag` | Complete RAG pipeline with indexing and querying |
| `chunking_strategies` | Comparison of different chunking approaches |
| `hybrid_search` | Dense + sparse hybrid retrieval with fusion |
| `metrics_evaluation` | Retrieval quality metrics (Recall, MRR, NDCG) |

## Feature Flags

Currently all features are enabled by default. Future releases may include optional features for different embedding backends.
