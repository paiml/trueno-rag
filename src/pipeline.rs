//! RAG Pipeline implementation with context assembly

use crate::{
    chunk::{Chunk, Chunker, RecursiveChunker},
    embed::{Embedder, MockEmbedder},
    fusion::FusionStrategy,
    index::{BM25Index, VectorStore},
    rerank::{NoOpReranker, Reranker},
    retrieve::{HybridRetriever, HybridRetrieverConfig, RetrievalResult},
    Document, DocumentId, Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Citation for a retrieved chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Citation ID (1-indexed for display)
    pub id: usize,
    /// Source document ID
    pub document_id: DocumentId,
    /// Source chunk ID
    pub chunk_id: crate::ChunkId,
    /// Document title (if available)
    pub title: Option<String>,
    /// Source URL (if available)
    pub url: Option<String>,
    /// Page number (if available)
    pub page: Option<usize>,
}

/// A chunk in the assembled context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextChunk {
    /// The chunk content
    pub content: String,
    /// Citation ID
    pub citation_id: usize,
    /// Retrieval score
    pub retrieval_score: f32,
    /// Rerank score (if available)
    pub rerank_score: Option<f32>,
}

/// Assembled context from retrieval results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembledContext {
    /// Ordered chunks in context
    pub chunks: Vec<ContextChunk>,
    /// Total token count (estimated)
    pub total_tokens: usize,
    /// Source citations
    pub citations: Vec<Citation>,
}

impl AssembledContext {
    /// Create a new empty context
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            total_tokens: 0,
            citations: Vec::new(),
        }
    }

    /// Add a chunk to the context
    pub fn add_chunk(&mut self, result: &RetrievalResult, citation_id: usize) {
        let chunk = ContextChunk {
            content: result.chunk.content.clone(),
            citation_id,
            retrieval_score: result.best_score(),
            rerank_score: result.rerank_score,
        };

        // Estimate tokens (rough: ~4 chars per token for English)
        self.total_tokens += result.chunk.content.len() / 4;
        self.chunks.push(chunk);
    }

    /// Add a citation
    pub fn add_citation(&mut self, result: &RetrievalResult) -> usize {
        let id = self.citations.len() + 1;

        let citation = Citation {
            id,
            document_id: result.chunk.document_id,
            chunk_id: result.chunk.id,
            title: result.chunk.metadata.title.clone(),
            url: None, // Would come from document metadata
            page: result.chunk.metadata.page,
        };

        self.citations.push(citation);
        id
    }

    /// Format context with inline citations
    #[must_use]
    pub fn format_with_citations(&self) -> String {
        self.chunks
            .iter()
            .map(|c| format!("{} [{}]", c.content, c.citation_id))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Format context without citations
    #[must_use]
    pub fn format_plain(&self) -> String {
        self.chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Generate citation list
    #[must_use]
    pub fn citation_list(&self) -> String {
        self.citations
            .iter()
            .map(|c| {
                let title = c.title.as_deref().unwrap_or("Untitled");
                format!("[{}] {}", c.id, title)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get the number of chunks
    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Check if the context is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

impl Default for AssembledContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Strategy for assembling context from retrieval results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum AssemblyStrategy {
    /// Simple concatenation in rank order
    #[default]
    Sequential,
    /// Group by document, then by rank
    DocumentGrouped,
    /// Interleave chunks for diversity
    Interleaved,
}

/// Context assembler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAssemblerConfig {
    /// Maximum context length in tokens (estimated)
    pub max_tokens: usize,
    /// Assembly strategy
    pub strategy: AssemblyStrategy,
    /// Include citations
    pub include_citations: bool,
}

impl Default for ContextAssemblerConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            strategy: AssemblyStrategy::Sequential,
            include_citations: true,
        }
    }
}

/// Assembles retrieved chunks into a coherent context
#[derive(Debug, Clone)]
pub struct ContextAssembler {
    config: ContextAssemblerConfig,
}

impl ContextAssembler {
    /// Create a new context assembler
    #[must_use]
    pub fn new(config: ContextAssemblerConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self::new(ContextAssemblerConfig {
            max_tokens,
            ..Default::default()
        })
    }

    /// Assemble context from retrieval results
    #[must_use]
    pub fn assemble(&self, results: &[RetrievalResult]) -> AssembledContext {
        match self.config.strategy {
            AssemblyStrategy::Sequential => self.assemble_sequential(results),
            AssemblyStrategy::DocumentGrouped => self.assemble_grouped(results),
            AssemblyStrategy::Interleaved => self.assemble_interleaved(results),
        }
    }

    fn assemble_sequential(&self, results: &[RetrievalResult]) -> AssembledContext {
        let mut context = AssembledContext::new();
        let mut remaining_tokens = self.config.max_tokens;

        for result in results {
            let chunk_tokens = result.chunk.content.len() / 4; // Rough estimate

            if chunk_tokens > remaining_tokens {
                // Could truncate, but for now we just stop
                break;
            }

            let citation_id = if self.config.include_citations {
                context.add_citation(result)
            } else {
                0
            };

            context.add_chunk(result, citation_id);
            remaining_tokens = remaining_tokens.saturating_sub(chunk_tokens);
        }

        context
    }

    fn assemble_grouped(&self, results: &[RetrievalResult]) -> AssembledContext {
        // Group by document
        let mut by_doc: HashMap<DocumentId, Vec<&RetrievalResult>> = HashMap::new();
        for result in results {
            by_doc
                .entry(result.chunk.document_id)
                .or_default()
                .push(result);
        }

        // Flatten while respecting order within documents
        let mut context = AssembledContext::new();
        let mut remaining_tokens = self.config.max_tokens;

        for (_, doc_results) in by_doc {
            for result in doc_results {
                let chunk_tokens = result.chunk.content.len() / 4;

                if chunk_tokens > remaining_tokens {
                    break;
                }

                let citation_id = if self.config.include_citations {
                    context.add_citation(result)
                } else {
                    0
                };

                context.add_chunk(result, citation_id);
                remaining_tokens = remaining_tokens.saturating_sub(chunk_tokens);
            }
        }

        context
    }

    fn assemble_interleaved(&self, results: &[RetrievalResult]) -> AssembledContext {
        // For now, same as sequential but could implement round-robin from different docs
        self.assemble_sequential(results)
    }
}

impl Default for ContextAssembler {
    fn default() -> Self {
        Self::new(ContextAssemblerConfig::default())
    }
}

/// RAG Pipeline configuration
#[derive(Debug, Clone)]
pub struct RagPipelineConfig {
    /// Chunking chunk size
    pub chunk_size: usize,
    /// Chunking overlap
    pub chunk_overlap: usize,
    /// Embedding dimension
    pub embedding_dimension: usize,
    /// Retrieval config
    pub retrieval: HybridRetrieverConfig,
    /// Context assembly config
    pub context: ContextAssemblerConfig,
}

impl Default for RagPipelineConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            embedding_dimension: 384,
            retrieval: HybridRetrieverConfig::default(),
            context: ContextAssemblerConfig::default(),
        }
    }
}

/// Complete RAG pipeline
pub struct RagPipeline<E: Embedder, R: Reranker> {
    /// Document chunker
    chunker: Box<dyn Chunker>,
    /// Embedder
    embedder: E,
    /// Hybrid retriever
    retriever: HybridRetriever<E>,
    /// Reranker
    reranker: R,
    /// Context assembler
    assembler: ContextAssembler,
    /// Indexed document count
    document_count: usize,
}

impl<E: Embedder + Clone, R: Reranker> RagPipeline<E, R> {
    /// Index a single document
    pub fn index_document(&mut self, document: &Document) -> Result<Vec<Chunk>> {
        // Chunk the document
        let mut chunks = self.chunker.chunk(document)?;

        // Embed the chunks
        self.embedder.embed_chunks(&mut chunks)?;

        // Add to retriever (both dense and sparse indices)
        for chunk in &chunks {
            self.retriever.index(chunk.clone())?;
        }

        self.document_count += 1;
        Ok(chunks)
    }

    /// Index multiple documents
    pub fn index_documents(&mut self, documents: &[Document]) -> Result<usize> {
        let mut total_chunks = 0;
        for doc in documents {
            let chunks = self.index_document(doc)?;
            total_chunks += chunks.len();
        }
        Ok(total_chunks)
    }

    /// Get the number of indexed documents
    #[must_use]
    pub fn document_count(&self) -> usize {
        self.document_count
    }

    /// Get the number of indexed chunks
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.retriever.len()
    }

    /// Query the pipeline
    pub fn query(&self, query: &str, k: usize) -> Result<Vec<RetrievalResult>> {
        // Retrieve
        let mut results = self.retriever.retrieve(query, k * 2)?;

        // Rerank
        results = self.reranker.rerank(query, &results, k)?;

        Ok(results)
    }

    /// Query and assemble context
    pub fn query_with_context(
        &self,
        query: &str,
        k: usize,
    ) -> Result<(Vec<RetrievalResult>, AssembledContext)> {
        let results = self.query(query, k)?;
        let context = self.assembler.assemble(&results);
        Ok((results, context))
    }

    /// Get the context assembler
    #[must_use]
    pub fn assembler(&self) -> &ContextAssembler {
        &self.assembler
    }

    /// Assemble context from results
    #[must_use]
    pub fn assemble_context(&self, results: &[RetrievalResult]) -> AssembledContext {
        self.assembler.assemble(results)
    }

    /// Get the chunker
    #[must_use]
    pub fn chunker(&self) -> &dyn Chunker {
        self.chunker.as_ref()
    }

    /// Get the embedder
    #[must_use]
    pub fn embedder(&self) -> &E {
        &self.embedder
    }
}

/// Builder for RAG pipeline
pub struct RagPipelineBuilder<E: Embedder, R: Reranker> {
    chunker: Option<Box<dyn Chunker>>,
    embedder: Option<E>,
    vector_store: Option<VectorStore>,
    sparse_index: Option<BM25Index>,
    reranker: Option<R>,
    fusion: FusionStrategy,
    assembler_config: ContextAssemblerConfig,
}

impl<E: Embedder + Clone, R: Reranker> RagPipelineBuilder<E, R> {
    /// Create a new pipeline builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunker: None,
            embedder: None,
            vector_store: None,
            sparse_index: None,
            reranker: None,
            fusion: FusionStrategy::default(),
            assembler_config: ContextAssemblerConfig::default(),
        }
    }

    /// Set the chunker
    #[must_use]
    pub fn chunker(mut self, chunker: impl Chunker + 'static) -> Self {
        self.chunker = Some(Box::new(chunker));
        self
    }

    /// Set the embedder
    #[must_use]
    pub fn embedder(mut self, embedder: E) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set the vector store
    #[must_use]
    pub fn vector_store(mut self, store: VectorStore) -> Self {
        self.vector_store = Some(store);
        self
    }

    /// Set the sparse index
    #[must_use]
    pub fn sparse_index(mut self, index: BM25Index) -> Self {
        self.sparse_index = Some(index);
        self
    }

    /// Set the reranker
    #[must_use]
    pub fn reranker(mut self, reranker: R) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Set the fusion strategy
    #[must_use]
    pub fn fusion(mut self, fusion: FusionStrategy) -> Self {
        self.fusion = fusion;
        self
    }

    /// Set max context tokens
    #[must_use]
    pub fn max_context_tokens(mut self, max_tokens: usize) -> Self {
        self.assembler_config.max_tokens = max_tokens;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Result<RagPipeline<E, R>> {
        let embedder = self
            .embedder
            .ok_or_else(|| Error::InvalidConfig("embedder required".to_string()))?;

        let reranker = self
            .reranker
            .ok_or_else(|| Error::InvalidConfig("reranker required".to_string()))?;

        let chunker = self
            .chunker
            .unwrap_or_else(|| Box::new(RecursiveChunker::new(512, 50)));

        let vector_store = self
            .vector_store
            .unwrap_or_else(|| VectorStore::with_dimension(embedder.dimension()));

        let sparse_index = self.sparse_index.unwrap_or_default();

        let retrieval_config = HybridRetrieverConfig {
            fusion: self.fusion,
            ..Default::default()
        };

        let retriever = HybridRetriever::new(vector_store, sparse_index, embedder.clone())
            .with_config(retrieval_config);

        let assembler = ContextAssembler::new(self.assembler_config);

        Ok(RagPipeline {
            chunker,
            embedder,
            retriever,
            reranker,
            assembler,
            document_count: 0,
        })
    }
}

impl<E: Embedder + Clone, R: Reranker> Default for RagPipelineBuilder<E, R> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified pipeline builder with defaults
#[must_use]
pub fn pipeline_builder() -> RagPipelineBuilder<MockEmbedder, NoOpReranker> {
    RagPipelineBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use crate::embed::MockEmbedder;
    use crate::rerank::LexicalReranker;

    fn create_result(content: &str, score: f32) -> RetrievalResult {
        let chunk = Chunk::new(DocumentId::new(), content.to_string(), 0, content.len());
        RetrievalResult::new(chunk).with_fused_score(score)
    }

    fn create_result_with_title(content: &str, title: &str, score: f32) -> RetrievalResult {
        let mut chunk = Chunk::new(DocumentId::new(), content.to_string(), 0, content.len());
        chunk.metadata.title = Some(title.to_string());
        RetrievalResult::new(chunk).with_fused_score(score)
    }

    // ============ Citation Tests ============

    #[test]
    fn test_citation_creation() {
        let result = create_result_with_title("content", "Test Doc", 0.9);
        let mut context = AssembledContext::new();
        let id = context.add_citation(&result);

        assert_eq!(id, 1);
        assert_eq!(context.citations.len(), 1);
        assert_eq!(context.citations[0].title, Some("Test Doc".to_string()));
    }

    // ============ AssembledContext Tests ============

    #[test]
    fn test_assembled_context_new() {
        let context = AssembledContext::new();
        assert!(context.is_empty());
        assert_eq!(context.len(), 0);
        assert_eq!(context.total_tokens, 0);
    }

    #[test]
    fn test_assembled_context_add_chunk() {
        let mut context = AssembledContext::new();
        let result = create_result("Test content here", 0.9);

        context.add_chunk(&result, 1);

        assert_eq!(context.len(), 1);
        assert!(context.total_tokens > 0);
    }

    #[test]
    fn test_assembled_context_format_with_citations() {
        let mut context = AssembledContext::new();
        let result1 = create_result("First chunk", 0.9);
        let result2 = create_result("Second chunk", 0.8);

        let id1 = context.add_citation(&result1);
        context.add_chunk(&result1, id1);
        let id2 = context.add_citation(&result2);
        context.add_chunk(&result2, id2);

        let formatted = context.format_with_citations();

        assert!(formatted.contains("[1]"));
        assert!(formatted.contains("[2]"));
        assert!(formatted.contains("First chunk"));
        assert!(formatted.contains("Second chunk"));
    }

    #[test]
    fn test_assembled_context_format_plain() {
        let mut context = AssembledContext::new();
        let result = create_result("Plain content", 0.9);
        context.add_chunk(&result, 1);

        let formatted = context.format_plain();
        assert_eq!(formatted, "Plain content");
        assert!(!formatted.contains('['));
    }

    #[test]
    fn test_assembled_context_citation_list() {
        let mut context = AssembledContext::new();
        let result1 = create_result_with_title("content", "Doc A", 0.9);
        let result2 = create_result_with_title("content", "Doc B", 0.8);

        context.add_citation(&result1);
        context.add_citation(&result2);

        let list = context.citation_list();
        assert!(list.contains("[1] Doc A"));
        assert!(list.contains("[2] Doc B"));
    }

    // ============ ContextAssembler Tests ============

    #[test]
    fn test_context_assembler_default() {
        let assembler = ContextAssembler::default();
        assert_eq!(assembler.config.max_tokens, 4096);
    }

    #[test]
    fn test_context_assembler_with_max_tokens() {
        let assembler = ContextAssembler::with_max_tokens(2048);
        assert_eq!(assembler.config.max_tokens, 2048);
    }

    #[test]
    fn test_context_assembler_sequential() {
        let assembler = ContextAssembler::default();
        let results = vec![
            create_result("First", 0.9),
            create_result("Second", 0.8),
            create_result("Third", 0.7),
        ];

        let context = assembler.assemble(&results);

        assert_eq!(context.len(), 3);
        assert_eq!(context.citations.len(), 3);
    }

    #[test]
    fn test_context_assembler_max_tokens() {
        let assembler = ContextAssembler::with_max_tokens(10); // Very small

        let results = vec![
            create_result("A".repeat(100).as_str(), 0.9),
            create_result("B".repeat(100).as_str(), 0.8),
        ];

        let context = assembler.assemble(&results);

        // Should have stopped due to token limit
        assert!(context.len() < 2);
    }

    #[test]
    fn test_context_assembler_no_citations() {
        let config = ContextAssemblerConfig {
            include_citations: false,
            ..Default::default()
        };
        let assembler = ContextAssembler::new(config);

        let results = vec![create_result("Content", 0.9)];
        let context = assembler.assemble(&results);

        // Citations still tracked but IDs are 0
        assert_eq!(context.chunks[0].citation_id, 0);
    }

    // ============ RagPipelineBuilder Tests ============

    #[test]
    fn test_pipeline_builder_new() {
        let builder: RagPipelineBuilder<MockEmbedder, NoOpReranker> = RagPipelineBuilder::new();
        // Should compile and create
        let _ = builder;
    }

    #[test]
    fn test_pipeline_builder_missing_embedder() {
        let builder: RagPipelineBuilder<MockEmbedder, NoOpReranker> =
            RagPipelineBuilder::new().reranker(NoOpReranker::new());

        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_builder_missing_reranker() {
        let builder: RagPipelineBuilder<MockEmbedder, NoOpReranker> =
            RagPipelineBuilder::new().embedder(MockEmbedder::new(64));

        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_builder_complete() {
        let builder = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .chunker(RecursiveChunker::new(256, 32))
            .fusion(FusionStrategy::RRF { k: 60.0 })
            .max_context_tokens(2048);

        let pipeline = builder.build().unwrap();
        assert_eq!(pipeline.assembler.config.max_tokens, 2048);
    }

    #[test]
    fn test_pipeline_builder_with_lexical_reranker() {
        let pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(LexicalReranker::new())
            .build()
            .unwrap();

        let _ = pipeline;
    }

    // ============ Full Pipeline Tests ============

    #[test]
    fn test_pipeline_query_empty() {
        let pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .build()
            .unwrap();

        let results = pipeline.query("test query", 10).unwrap();
        assert!(results.is_empty()); // Nothing indexed
    }

    // ============ Integration Tests ============

    #[test]
    fn test_full_pipeline_index_and_query() {
        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(LexicalReranker::new())
            .chunker(RecursiveChunker::new(100, 20))
            .build()
            .unwrap();

        // Index documents
        let doc1 = Document::new(
            "Machine learning is a subset of artificial intelligence. It enables computers to learn from data.",
        )
        .with_title("ML Intro");

        let doc2 = Document::new(
            "Deep learning uses neural networks with many layers. It excels at image recognition.",
        )
        .with_title("Deep Learning");

        pipeline.index_document(&doc1).unwrap();
        pipeline.index_document(&doc2).unwrap();

        assert_eq!(pipeline.document_count(), 2);
        assert!(pipeline.chunk_count() >= 2);

        // Query
        let results = pipeline.query("machine learning", 5).unwrap();
        assert!(!results.is_empty());

        // Context assembly
        let (results, context) = pipeline.query_with_context("machine learning", 5).unwrap();
        assert!(!results.is_empty());
        assert!(!context.is_empty());
        assert!(!context.citations.is_empty());
    }

    #[test]
    fn test_pipeline_with_hybrid_retrieval() {
        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(LexicalReranker::new())
            .fusion(FusionStrategy::RRF { k: 60.0 })
            .build()
            .unwrap();

        let doc = Document::new(
            "Rust is a systems programming language. It focuses on safety and performance.",
        );
        pipeline.index_document(&doc).unwrap();

        // Should work with both dense and sparse retrieval
        let results = pipeline.query("rust programming", 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_pipeline_context_with_citations() {
        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .max_context_tokens(2000)
            .build()
            .unwrap();

        let doc = Document::new("Test content for citation tracking.").with_title("Test Document");
        pipeline.index_document(&doc).unwrap();

        let (_, context) = pipeline.query_with_context("test content", 5).unwrap();

        let formatted = context.format_with_citations();
        assert!(formatted.contains("[1]"));

        let citation_list = context.citation_list();
        assert!(citation_list.contains("Test Document"));
    }

    #[test]
    fn test_pipeline_multiple_documents() {
        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(LexicalReranker::new())
            .build()
            .unwrap();

        let docs = vec![
            Document::new("Document about cats and their behavior."),
            Document::new("Document about dogs and training."),
            Document::new("Document about birds and migration."),
        ];

        let chunk_count = pipeline.index_documents(&docs).unwrap();
        assert!(chunk_count >= 3);
        assert_eq!(pipeline.document_count(), 3);

        let results = pipeline.query("cats", 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_pipeline_empty_query_result() {
        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .build()
            .unwrap();

        pipeline
            .index_document(&Document::new("Some content here."))
            .unwrap();

        // Query with no matching terms
        let results = pipeline.query("xyz123nonexistent", 5).unwrap();
        // Results may be empty or contain low-scored results depending on implementation
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_pipeline_reranker_effect() {
        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(LexicalReranker::new().with_weights(0.5, 0.3, 0.2))
            .build()
            .unwrap();

        pipeline
            .index_document(&Document::new("exact phrase machine learning here"))
            .unwrap();
        pipeline
            .index_document(&Document::new("machine related and learning separate"))
            .unwrap();

        let results = pipeline.query("machine learning", 5).unwrap();
        assert!(!results.is_empty());
        // Reranker should have assigned scores
        for result in &results {
            if result.rerank_score.is_some() {
                assert!(result.rerank_score.unwrap() >= 0.0);
            }
        }
    }

    #[test]
    fn test_pipeline_fusion_strategies() {
        for fusion in [
            FusionStrategy::RRF { k: 60.0 },
            FusionStrategy::Linear { dense_weight: 0.7 },
            FusionStrategy::DBSF,
            FusionStrategy::Union,
            FusionStrategy::Intersection,
        ] {
            let mut pipeline = RagPipelineBuilder::new()
                .embedder(MockEmbedder::new(64))
                .reranker(NoOpReranker::new())
                .fusion(fusion)
                .build()
                .unwrap();

            pipeline
                .index_document(&Document::new("Test document for fusion strategies."))
                .unwrap();

            let results = pipeline.query("test document", 5).unwrap();
            // Pipeline should work with all fusion strategies
            assert!(results.len() <= 5);
        }
    }

    // ============ Additional Coverage Tests ============

    #[test]
    fn test_context_assembler_grouped_strategy() {
        let config = ContextAssemblerConfig {
            strategy: AssemblyStrategy::DocumentGrouped,
            ..Default::default()
        };
        let assembler = ContextAssembler::new(config);

        // Create results from different documents
        let doc1 = DocumentId::new();
        let doc2 = DocumentId::new();

        let chunk1 = Chunk::new(doc1, "Doc1 Chunk1".to_string(), 0, 11);
        let chunk2 = Chunk::new(doc1, "Doc1 Chunk2".to_string(), 12, 23);
        let chunk3 = Chunk::new(doc2, "Doc2 Chunk1".to_string(), 0, 11);

        let results = vec![
            RetrievalResult::new(chunk1).with_fused_score(0.9),
            RetrievalResult::new(chunk3).with_fused_score(0.8),
            RetrievalResult::new(chunk2).with_fused_score(0.7),
        ];

        let context = assembler.assemble(&results);

        // Should have all 3 chunks
        assert_eq!(context.len(), 3);
        assert_eq!(context.citations.len(), 3);
    }

    #[test]
    fn test_context_assembler_interleaved_strategy() {
        let config = ContextAssemblerConfig {
            strategy: AssemblyStrategy::Interleaved,
            ..Default::default()
        };
        let assembler = ContextAssembler::new(config);

        let results = vec![create_result("Chunk A", 0.9), create_result("Chunk B", 0.8)];

        let context = assembler.assemble(&results);

        assert_eq!(context.len(), 2);
    }

    #[test]
    fn test_context_assembler_grouped_max_tokens() {
        let config = ContextAssemblerConfig {
            max_tokens: 10, // Very small
            strategy: AssemblyStrategy::DocumentGrouped,
            include_citations: true,
        };
        let assembler = ContextAssembler::new(config);

        let results = vec![
            create_result(&"A".repeat(100), 0.9),
            create_result(&"B".repeat(100), 0.8),
        ];

        let context = assembler.assemble(&results);

        // Should have stopped due to token limit
        assert!(context.len() < 2);
    }

    #[test]
    fn test_context_assembler_grouped_no_citations() {
        let config = ContextAssemblerConfig {
            strategy: AssemblyStrategy::DocumentGrouped,
            include_citations: false,
            ..Default::default()
        };
        let assembler = ContextAssembler::new(config);

        let results = vec![create_result("Content", 0.9)];
        let context = assembler.assemble(&results);

        assert_eq!(context.chunks[0].citation_id, 0);
    }

    #[test]
    fn test_pipeline_builder_with_vector_store() {
        let custom_store = VectorStore::with_dimension(128);

        let pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(128))
            .reranker(NoOpReranker::new())
            .vector_store(custom_store)
            .build()
            .unwrap();

        assert_eq!(pipeline.document_count(), 0);
    }

    #[test]
    fn test_pipeline_builder_with_sparse_index() {
        let custom_index = BM25Index::with_params(1.5, 0.8);

        let pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .sparse_index(custom_index)
            .build()
            .unwrap();

        assert_eq!(pipeline.chunk_count(), 0);
    }

    #[test]
    fn test_pipeline_builder_function() {
        // Test the simplified pipeline_builder() function
        let builder = pipeline_builder();
        let pipeline = builder
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .build()
            .unwrap();

        assert_eq!(pipeline.document_count(), 0);
    }

    #[test]
    fn test_pipeline_chunker_method() {
        let pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .chunker(RecursiveChunker::new(256, 32))
            .build()
            .unwrap();

        // Access chunker through public method
        let chunker = pipeline.chunker();
        let doc = Document::new("Test document content for chunking.");
        let estimate = chunker.estimate_chunks(&doc);
        assert!(estimate >= 1);
    }

    #[test]
    fn test_pipeline_embedder_method() {
        let pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(128))
            .reranker(NoOpReranker::new())
            .build()
            .unwrap();

        // Access embedder through public method
        let embedder = pipeline.embedder();
        assert_eq!(embedder.dimension(), 128);
    }

    #[test]
    fn test_pipeline_assemble_context_method() {
        let mut pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .build()
            .unwrap();

        pipeline
            .index_document(&Document::new("Test document content."))
            .unwrap();

        let results = pipeline.query("test", 5).unwrap();

        // Use public assemble_context method
        let context = pipeline.assemble_context(&results);
        assert!(context.len() <= results.len());
    }

    #[test]
    fn test_pipeline_assembler_method() {
        let pipeline = RagPipelineBuilder::new()
            .embedder(MockEmbedder::new(64))
            .reranker(NoOpReranker::new())
            .max_context_tokens(1000)
            .build()
            .unwrap();

        // Access assembler through public method
        let assembler = pipeline.assembler();
        assert_eq!(assembler.config.max_tokens, 1000);
    }

    #[test]
    fn test_assembled_context_with_page_metadata() {
        let mut context = AssembledContext::new();

        let mut chunk = Chunk::new(DocumentId::new(), "Page content".to_string(), 0, 12);
        chunk.metadata.page = Some(5);
        chunk.metadata.title = Some("Document Title".to_string());

        let result = RetrievalResult::new(chunk).with_fused_score(0.9);
        let id = context.add_citation(&result);
        context.add_chunk(&result, id);

        assert_eq!(context.citations[0].page, Some(5));
        assert_eq!(
            context.citations[0].title,
            Some("Document Title".to_string())
        );
    }

    #[test]
    fn test_citation_without_title_uses_untitled() {
        let mut context = AssembledContext::new();

        // Create result without title
        let chunk = Chunk::new(DocumentId::new(), "content".to_string(), 0, 7);
        let result = RetrievalResult::new(chunk).with_fused_score(0.9);

        context.add_citation(&result);
        let list = context.citation_list();

        assert!(list.contains("Untitled"));
    }

    #[test]
    fn test_rag_pipeline_config_default() {
        let config = RagPipelineConfig::default();

        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.chunk_overlap, 50);
        assert_eq!(config.embedding_dimension, 384);
        assert_eq!(config.context.max_tokens, 4096);
    }

    #[test]
    fn test_assembly_strategy_serialization() {
        let strategies = vec![
            AssemblyStrategy::Sequential,
            AssemblyStrategy::DocumentGrouped,
            AssemblyStrategy::Interleaved,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let deserialized: AssemblyStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(
                std::mem::discriminant(&strategy),
                std::mem::discriminant(&deserialized)
            );
        }
    }

    #[test]
    fn test_context_assembler_config_serialization() {
        let config = ContextAssemblerConfig {
            max_tokens: 2048,
            strategy: AssemblyStrategy::DocumentGrouped,
            include_citations: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ContextAssemblerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.max_tokens, deserialized.max_tokens);
        assert!(!deserialized.include_citations);
    }

    #[test]
    fn test_citation_serialization() {
        let citation = Citation {
            id: 1,
            document_id: DocumentId::new(),
            chunk_id: crate::ChunkId::new(),
            title: Some("Test".to_string()),
            url: Some("https://example.com".to_string()),
            page: Some(10),
        };

        let json = serde_json::to_string(&citation).unwrap();
        let deserialized: Citation = serde_json::from_str(&json).unwrap();

        assert_eq!(citation.id, deserialized.id);
        assert_eq!(citation.title, deserialized.title);
        assert_eq!(citation.url, deserialized.url);
        assert_eq!(citation.page, deserialized.page);
    }

    #[test]
    fn test_context_chunk_serialization() {
        let chunk = ContextChunk {
            content: "Test content".to_string(),
            citation_id: 1,
            retrieval_score: 0.9,
            rerank_score: Some(0.95),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: ContextChunk = serde_json::from_str(&json).unwrap();

        assert_eq!(chunk.content, deserialized.content);
        assert_eq!(chunk.citation_id, deserialized.citation_id);
    }

    #[test]
    fn test_assembled_context_serialization() {
        let mut context = AssembledContext::new();
        let result = create_result("Test", 0.9);
        let id = context.add_citation(&result);
        context.add_chunk(&result, id);

        let json = serde_json::to_string(&context).unwrap();
        let deserialized: AssembledContext = serde_json::from_str(&json).unwrap();

        assert_eq!(context.len(), deserialized.len());
        assert_eq!(context.citations.len(), deserialized.citations.len());
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_assembled_context_tokens_non_negative(
            content in "[a-zA-Z ]{10,100}"
        ) {
            let mut context = AssembledContext::new();
            let result = create_result(&content, 0.9);
            context.add_chunk(&result, 1);

            prop_assert!(context.total_tokens > 0);
        }

        #[test]
        fn prop_citation_ids_sequential(n in 1usize..10) {
            let mut context = AssembledContext::new();

            for i in 0..n {
                let result = create_result(&format!("content {i}"), 0.9);
                let id = context.add_citation(&result);
                prop_assert_eq!(id, i + 1);
            }
        }

        #[test]
        fn prop_assembler_respects_max_tokens(
            max_tokens in 100usize..1000,
            n_chunks in 1usize..10
        ) {
            let assembler = ContextAssembler::with_max_tokens(max_tokens);

            let results: Vec<_> = (0..n_chunks)
                .map(|i| create_result(&format!("chunk content {i}"), 1.0 - i as f32 * 0.1))
                .collect();

            let context = assembler.assemble(&results);

            // Total tokens should not exceed max (with some tolerance for estimation)
            prop_assert!(context.total_tokens <= max_tokens + 50);
        }
    }
}
