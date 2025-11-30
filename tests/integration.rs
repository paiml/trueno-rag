//! Integration tests for trueno-rag

use trueno_rag::{
    chunk::{Chunker, ParagraphChunker, RecursiveChunker, SentenceChunker, StructuralChunker},
    embed::MockEmbedder,
    fusion::FusionStrategy,
    pipeline::RagPipelineBuilder,
    rerank::{LexicalReranker, NoOpReranker},
    Document,
};

#[test]
fn test_end_to_end_rag_pipeline() {
    let mut pipeline = RagPipelineBuilder::new()
        .chunker(RecursiveChunker::new(200, 20))
        .embedder(MockEmbedder::new(128))
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .max_context_tokens(2000)
        .build()
        .expect("Failed to build pipeline");

    // Index multiple documents
    let docs = vec![
        Document::new(
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             It achieves memory safety without garbage collection.",
        )
        .with_title("Rust Overview"),
        Document::new(
            "Python is a high-level programming language known for its readability and \
             extensive standard library. It's popular for data science and web development.",
        )
        .with_title("Python Overview"),
        Document::new(
            "Machine learning is a subset of artificial intelligence that enables systems \
             to learn and improve from experience without being explicitly programmed.",
        )
        .with_title("ML Introduction"),
    ];

    let chunk_count = pipeline.index_documents(&docs).expect("Failed to index");
    assert!(chunk_count >= 3);
    assert_eq!(pipeline.document_count(), 3);

    // Query for Rust-related content
    let (results, context) = pipeline
        .query_with_context("memory safety in systems programming", 5)
        .expect("Query failed");

    assert!(!results.is_empty());
    assert!(!context.is_empty());

    // The top result should be about Rust
    let top_content = &results[0].chunk.content.to_lowercase();
    assert!(
        top_content.contains("rust") || top_content.contains("memory"),
        "Expected Rust-related content in top result"
    );
}

#[test]
fn test_different_chunking_strategies() {
    let doc = Document::new(
        "First paragraph about topic A.\n\n\
         Second paragraph about topic B.\n\n\
         Third paragraph about topic C.",
    );

    // Test ParagraphChunker
    let para_chunker = ParagraphChunker::new(1);
    let para_chunks = para_chunker.chunk(&doc).expect("ParagraphChunker failed");
    assert_eq!(para_chunks.len(), 3);

    // Test SentenceChunker
    let sent_chunker = SentenceChunker::new(2, 0);
    let sent_chunks = sent_chunker.chunk(&doc).expect("SentenceChunker failed");
    assert!(!sent_chunks.is_empty());

    // Test StructuralChunker with markdown
    let md_doc = Document::new(
        "# Header 1\n\nContent 1.\n\n# Header 2\n\nContent 2.",
    );
    let struct_chunker = StructuralChunker::new(true, 500);
    let struct_chunks = struct_chunker.chunk(&md_doc).expect("StructuralChunker failed");
    assert_eq!(struct_chunks.len(), 2);
}

#[test]
fn test_fusion_strategies_produce_results() {
    let strategies = vec![
        FusionStrategy::RRF { k: 60.0 },
        FusionStrategy::Linear { dense_weight: 0.7 },
        FusionStrategy::DBSF,
        FusionStrategy::Union,
    ];

    for strategy in strategies {
        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .fusion(strategy.clone())
            .build()
            .expect("Failed to build pipeline");

        pipeline
            .index_document(&Document::new("Test document content here."))
            .expect("Failed to index");

        let results = pipeline.query("test", 5).expect("Query failed");
        assert!(
            results.len() <= 5,
            "Strategy {:?} returned too many results",
            strategy
        );
    }
}

#[test]
fn test_context_assembly_with_citations() {
    let mut pipeline = RagPipelineBuilder::new()
        .embedder(MockEmbedder::new(64))
        .reranker(NoOpReranker::new())
        .max_context_tokens(1000)
        .build()
        .expect("Failed to build pipeline");

    let doc = Document::new("Important content for citation.")
        .with_title("Test Document");
    pipeline.index_document(&doc).expect("Failed to index");

    let (_, context) = pipeline
        .query_with_context("important content", 5)
        .expect("Query failed");

    let formatted = context.format_with_citations();
    assert!(formatted.contains("[1]"), "Expected citation marker");

    let citation_list = context.citation_list();
    assert!(
        citation_list.contains("Test Document"),
        "Expected document title in citations"
    );
}

#[test]
fn test_empty_document_handling() {
    let chunker = RecursiveChunker::new(100, 10);
    let empty_doc = Document::new("");

    let result = chunker.chunk(&empty_doc);
    assert!(result.is_err(), "Empty document should produce error");
}

#[test]
fn test_large_document_chunking() {
    let large_content = "This is a test sentence. ".repeat(1000);
    let doc = Document::new(large_content);

    let chunker = RecursiveChunker::new(500, 50);
    let chunks = chunker.chunk(&doc).expect("Chunking failed");

    // Should produce multiple chunks
    assert!(chunks.len() > 1, "Large document should produce multiple chunks");

    // Each chunk should be within size limit (with some tolerance)
    for chunk in &chunks {
        assert!(
            chunk.content.len() <= 600,
            "Chunk exceeds size limit: {} chars",
            chunk.content.len()
        );
    }
}

#[test]
fn test_query_ranking_consistency() {
    let mut pipeline = RagPipelineBuilder::new()
        .embedder(MockEmbedder::new(64))
        .reranker(LexicalReranker::new())
        .build()
        .expect("Failed to build pipeline");

    // Index documents with different relevance
    pipeline
        .index_document(&Document::new("exact match query terms here"))
        .expect("Failed to index");
    pipeline
        .index_document(&Document::new("completely unrelated content"))
        .expect("Failed to index");

    let results = pipeline
        .query("exact match query", 5)
        .expect("Query failed");

    if results.len() >= 2 {
        // First result should have higher score than second
        assert!(
            results[0].best_score() >= results[1].best_score(),
            "Results should be sorted by score"
        );
    }
}
