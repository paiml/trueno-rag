# Trueno-RAG Makefile
# Certeza Methodology - Tiered Quality Gates

SHELL := /bin/bash
.SUFFIXES:
.DELETE_ON_ERROR:

.PHONY: all build test test-fast lint fmt fmt-check coverage coverage-open bench clean doc check ci examples book

# Default target
all: test-fast lint

# Build
build:
	cargo build

# ============================================================================
# TEST TARGETS
# ============================================================================

# Fast tests (<30s): Uses nextest for parallelism if available
test-fast: ## Fast unit tests (<30s target)
	@echo "⚡ Running fast tests (target: <30s)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run --lib \
			--status-level skip \
			--failure-output immediate; \
	else \
		echo "💡 Install cargo-nextest for faster tests: cargo install cargo-nextest"; \
		time cargo test --lib; \
	fi
	@echo "✅ Fast tests passed"

# Standard tests
test: ## Standard tests
	@echo "🧪 Running standard tests..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		time cargo nextest run \
			--status-level skip \
			--failure-output immediate; \
	else \
		time cargo test; \
	fi
	@echo "✅ Standard tests passed"

# Linting
lint: ## Run clippy linter
	cargo clippy -- -D warnings

# Format
fmt: ## Format code
	cargo fmt

fmt-check: ## Check formatting
	cargo fmt --check

# ============================================================================
# COVERAGE TARGETS (Two-Phase Pattern from aprender/bashrs)
# ============================================================================
# CRITICAL: mold linker breaks LLVM coverage instrumentation
# Solution: Temporarily move ~/.cargo/config.toml during coverage runs

coverage: ## Generate HTML coverage report (target: <5 min)
	@echo "📊 Running coverage analysis (target: <5 min)..."
	@echo "🔍 Checking for cargo-llvm-cov and cargo-nextest..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@echo "⚙️  Temporarily disabling global cargo config (sccache/mold break coverage)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@echo "🧹 Cleaning old coverage data..."
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@echo "🧪 Phase 1: Running tests with instrumentation (no report)..."
	@cargo llvm-cov --no-report nextest --no-tests=warn --workspace --no-fail-fast --all-features
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@echo "⚙️  Restoring global cargo config..."
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 Reports:"
	@echo "- HTML: target/coverage/html/index.html"
	@echo "- LCOV: target/coverage/lcov.info"

coverage-open: ## Open HTML coverage report
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first"; \
	fi

# ============================================================================
# OTHER TARGETS
# ============================================================================

bench: ## Run benchmarks
	cargo bench

doc: ## Generate documentation
	cargo doc --no-deps --open

clean: ## Clean build artifacts
	cargo clean
	rm -rf target/coverage/

check: ## Quick check
	cargo check

ci: fmt-check lint test ## Full CI pipeline
	@echo "✅ All CI checks passed!"

examples: ## Run all examples
	@for example in examples/*.rs; do \
		name=$$(basename "$$example" .rs); \
		echo "Running $$name..."; \
		cargo run --example "$$name" --quiet 2>/dev/null || echo "  ⚠️ $$name failed"; \
	done

book: ## Build documentation book
	@command -v mdbook >/dev/null 2>&1 || cargo install mdbook
	cd book && mdbook build

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
