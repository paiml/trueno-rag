//! CLI Integration Tests (EXTREME TDD - RED Phase)
//!
//! These tests define the expected behavior BEFORE implementation.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Helper to get CLI command
fn cli() -> Command {
    Command::cargo_bin("trueno-rag").unwrap()
}

// ============================================================================
// INFO COMMAND TESTS
// ============================================================================

#[test]
fn test_info_shows_version() {
    cli()
        .arg("info")
        .assert()
        .success()
        .stdout(predicate::str::contains("Trueno-RAG Pipeline"))
        .stdout(predicate::str::contains("Version:"));
}

#[test]
fn test_info_shows_components() {
    cli()
        .arg("info")
        .assert()
        .success()
        .stdout(predicate::str::contains("Chunkers:"))
        .stdout(predicate::str::contains("Embedders:"))
        .stdout(predicate::str::contains("TF-IDF"));
}

// ============================================================================
// DEMO COMMAND TESTS
// ============================================================================

#[test]
fn test_demo_default_query() {
    cli()
        .arg("demo")
        .assert()
        .success()
        .stdout(predicate::str::contains("Trueno-RAG Demo"))
        .stdout(predicate::str::contains("Indexed"))
        .stdout(predicate::str::contains("Results"));
}

#[test]
fn test_demo_custom_query() {
    cli()
        .args(["demo", "--query", "What is deep learning?"])
        .assert()
        .success()
        .stdout(predicate::str::contains("What is deep learning?"));
}

#[test]
fn test_demo_custom_top_k() {
    cli()
        .args(["demo", "--top-k", "2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Results (2)").or(predicate::str::contains("Results (")));
}

#[test]
fn test_demo_shows_citations() {
    cli()
        .arg("demo")
        .assert()
        .success()
        .stdout(predicate::str::contains("Citations:"));
}

// ============================================================================
// INDEX COMMAND TESTS
// ============================================================================

#[test]
fn test_index_single_file() {
    let tmp = TempDir::new().unwrap();
    let doc_path = tmp.path().join("test.txt");
    fs::write(&doc_path, "This is a test document about machine learning.").unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            doc_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Indexed"));

    // Verify index was created
    assert!(index_path.exists(), "Index directory should be created");
}

#[test]
fn test_index_directory() {
    let tmp = TempDir::new().unwrap();
    let docs_dir = tmp.path().join("docs");
    fs::create_dir(&docs_dir).unwrap();

    fs::write(docs_dir.join("doc1.txt"), "First document about AI.").unwrap();
    fs::write(docs_dir.join("doc2.txt"), "Second document about ML.").unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            docs_dir.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("2 documents"));
}

#[test]
fn test_index_with_chunk_size() {
    let tmp = TempDir::new().unwrap();
    let doc_path = tmp.path().join("test.txt");
    fs::write(&doc_path, "A ".repeat(500)).unwrap(); // Long document

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            doc_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
            "--chunk-size",
            "100",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("chunks"));
}

#[test]
fn test_index_nonexistent_path_fails() {
    cli()
        .args(["index", "--path", "/nonexistent/path", "--output", "/tmp/out"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found").or(predicate::str::contains("No such file")));
}

// ============================================================================
// QUERY COMMAND TESTS
// ============================================================================

#[test]
fn test_query_requires_index() {
    let tmp = TempDir::new().unwrap();
    let index_path = tmp.path().join("nonexistent_index");

    cli()
        .args([
            "query",
            "test query",
            "--index",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("index").or(predicate::str::contains("not found")));
}

#[test]
fn test_query_with_index() {
    let tmp = TempDir::new().unwrap();

    // First, create an index
    let doc_path = tmp.path().join("test.txt");
    fs::write(
        &doc_path,
        "Machine learning is a field of artificial intelligence.",
    )
    .unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            doc_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Now query
    cli()
        .args([
            "query",
            "What is machine learning?",
            "--index",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Results"));
}

#[test]
fn test_query_json_output() {
    let tmp = TempDir::new().unwrap();

    // Create index
    let doc_path = tmp.path().join("test.txt");
    fs::write(&doc_path, "Test document content.").unwrap();

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            doc_path.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Query with JSON output
    cli()
        .args([
            "query",
            "test",
            "--index",
            index_path.to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("{").and(predicate::str::contains("}")));
}

#[test]
fn test_query_top_k() {
    let tmp = TempDir::new().unwrap();

    // Create index with multiple docs
    let docs_dir = tmp.path().join("docs");
    fs::create_dir(&docs_dir).unwrap();
    for i in 0..5 {
        fs::write(docs_dir.join(format!("doc{i}.txt")), format!("Document {i} content.")).unwrap();
    }

    let index_path = tmp.path().join("index");

    cli()
        .args([
            "index",
            "--path",
            docs_dir.to_str().unwrap(),
            "--output",
            index_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Query with limited results
    cli()
        .args([
            "query",
            "document",
            "--index",
            index_path.to_str().unwrap(),
            "--top-k",
            "2",
        ])
        .assert()
        .success();
}

// ============================================================================
// HELP AND VERSION TESTS
// ============================================================================

#[test]
fn test_help() {
    cli()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Pure-Rust RAG pipeline CLI"));
}

#[test]
fn test_version() {
    cli()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("trueno-rag"));
}

#[test]
fn test_subcommand_help() {
    cli()
        .args(["index", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Index documents"));
}
