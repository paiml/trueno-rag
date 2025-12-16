//! Property-based tests for trueno-rag

use proptest::prelude::*;
use trueno_rag::{
    chunk::{Chunker, FixedSizeChunker, ParagraphChunker, RecursiveChunker},
    embed::{cosine_similarity, Embedder, MockEmbedder},
    Document,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_recursive_chunker_produces_valid_chunks(
        content in "[a-zA-Z ]{100,1000}",
        chunk_size in 50usize..200,
        overlap in 0usize..50
    ) {
        let overlap = overlap.min(chunk_size / 2);
        let chunker = RecursiveChunker::new(chunk_size, overlap);
        let doc = Document::new(&content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            // All chunks should be non-empty
            for chunk in &chunks {
                prop_assert!(!chunk.content.is_empty());
            }

            // Chunk IDs should be unique
            let ids: std::collections::HashSet<_> = chunks.iter().map(|c| c.id).collect();
            prop_assert_eq!(ids.len(), chunks.len());
        }
    }

    #[test]
    fn prop_fixed_size_chunker_respects_size(
        content in "[a-zA-Z ]{200,500}",
        chunk_size in 50usize..150,
        overlap in 0usize..30
    ) {
        let overlap = overlap.min(chunk_size / 2);
        let chunker = FixedSizeChunker::new(chunk_size, overlap);
        let doc = Document::new(&content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            for chunk in &chunks {
                // Chunks should not exceed size by too much
                prop_assert!(chunk.content.len() <= chunk_size + 10);
            }
        }
    }

    #[test]
    fn prop_paragraph_chunker_groups_correctly(
        para_count in 2usize..8,
        max_paras in 1usize..4
    ) {
        // Create document with known paragraph count
        let content: String = (0..para_count)
            .map(|i| format!("Paragraph {} content here.", i))
            .collect::<Vec<_>>()
            .join("\n\n");

        let chunker = ParagraphChunker::new(max_paras);
        let doc = Document::new(&content);

        if let Ok(chunks) = chunker.chunk(&doc) {
            // Should have roughly ceil(para_count / max_paras) chunks
            let expected_min = (para_count + max_paras - 1) / max_paras;
            prop_assert!(chunks.len() >= expected_min.saturating_sub(1));
        }
    }

    #[test]
    fn prop_embedder_produces_consistent_dimension(
        text in "[a-zA-Z ]{10,100}",
        dimension in 32usize..512
    ) {
        let embedder = MockEmbedder::new(dimension);

        if let Ok(embedding) = embedder.embed(&text) {
            prop_assert_eq!(embedding.len(), dimension);
        }
    }

    #[test]
    fn prop_cosine_similarity_bounded(
        v1 in prop::collection::vec(-1.0f32..1.0, 10..50),
        v2 in prop::collection::vec(-1.0f32..1.0, 10..50)
    ) {
        if v1.len() == v2.len() {
            let sim = cosine_similarity(&v1, &v2);
            // Cosine similarity should be in [-1, 1] range
            prop_assert!(sim >= -1.1 && sim <= 1.1);
        }
    }

    #[test]
    fn prop_document_preserves_content(content in "[a-zA-Z0-9 ]{1,500}") {
        let doc = Document::new(&content);
        prop_assert_eq!(doc.content, content);
    }

    #[test]
    fn prop_document_with_metadata(
        content in "[a-zA-Z ]{10,100}",
        title in "[a-zA-Z ]{5,30}",
        source in "[a-zA-Z/:.]{10,50}"
    ) {
        let doc = Document::new(&content)
            .with_title(&title)
            .with_source(&source);

        prop_assert_eq!(doc.content, content);
        prop_assert_eq!(doc.title, Some(title));
        prop_assert_eq!(doc.source, Some(source));
    }
}

#[test]
fn test_chunker_estimate_accuracy() {
    let chunker = RecursiveChunker::new(100, 10);
    let doc = Document::new("Test content. ".repeat(50));

    let estimate = chunker.estimate_chunks(&doc);
    let actual = chunker.chunk(&doc).unwrap().len();

    // Estimate should be within 50% of actual
    let tolerance = (actual as f32 * 0.5).ceil() as usize;
    assert!(
        (estimate as i32 - actual as i32).unsigned_abs() as usize <= tolerance.max(2),
        "Estimate {} too far from actual {}",
        estimate,
        actual
    );
}
