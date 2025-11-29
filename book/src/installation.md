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

## Feature Flags

Currently all features are enabled by default. Future releases may include optional features for different embedding backends.
