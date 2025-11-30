//! Embedding generation for RAG pipelines

use crate::{Chunk, Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Pooling strategy for token embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Use [CLS] token embedding
    Cls,
    /// Mean of all token embeddings
    Mean,
    /// Mean with attention weighting
    WeightedMean,
    /// Last token (for decoder models)
    LastToken,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

/// Configuration for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Normalize embeddings to unit length
    pub normalize: bool,
    /// Instruction prefix for queries (asymmetric retrieval)
    pub query_prefix: Option<String>,
    /// Instruction prefix for documents
    pub document_prefix: Option<String>,
    /// Maximum sequence length in tokens
    pub max_length: usize,
    /// Pooling strategy
    pub pooling: PoolingStrategy,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            query_prefix: None,
            document_prefix: None,
            max_length: 512,
            pooling: PoolingStrategy::Mean,
        }
    }
}

/// Trait for embedding generation
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a single text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batch embed multiple texts
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Get model identifier
    fn model_id(&self) -> &str;

    /// Embed a query (may use query prefix)
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embed(query)
    }

    /// Embed a document (may use document prefix)
    fn embed_document(&self, document: &str) -> Result<Vec<f32>> {
        self.embed(document)
    }

    /// Embed chunks and update them in place
    fn embed_chunks(&self, chunks: &mut [Chunk]) -> Result<()> {
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = self.embed_batch(&texts)?;

        for (chunk, embedding) in chunks.iter_mut().zip(embeddings) {
            chunk.set_embedding(embedding);
        }

        Ok(())
    }
}

/// Mock embedder for testing (uses simple hash-based vectors)
#[derive(Debug, Clone)]
pub struct MockEmbedder {
    dimension: usize,
    model_id: String,
    config: EmbeddingConfig,
}

impl MockEmbedder {
    /// Create a new mock embedder
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            model_id: "mock-embedder".to_string(),
            config: EmbeddingConfig::default(),
        }
    }

    /// Set the model ID
    #[must_use]
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    /// Set configuration
    #[must_use]
    pub fn with_config(mut self, config: EmbeddingConfig) -> Self {
        self.config = config;
        self
    }

    fn hash_to_vector(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vector = Vec::with_capacity(self.dimension);
        let mut hasher = DefaultHasher::new();

        for i in 0..self.dimension {
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            // Convert hash to float in range [-1, 1]
            let value = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0;
            vector.push(value);
        }

        if self.config.normalize {
            Self::normalize_vector(&mut vector);
        }

        vector
    }

    fn normalize_vector(vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::EmptyDocument("empty text for embedding".to_string()));
        }

        let prefixed = if let Some(prefix) = &self.config.document_prefix {
            format!("{prefix}{text}")
        } else {
            text.to_string()
        };

        Ok(self.hash_to_vector(&prefixed))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if query.is_empty() {
            return Err(Error::Query("empty query".to_string()));
        }

        let prefixed = if let Some(prefix) = &self.config.query_prefix {
            format!("{prefix}{query}")
        } else {
            query.to_string()
        };

        Ok(self.hash_to_vector(&prefixed))
    }
}

/// TF-IDF based embedder (sparse-to-dense conversion)
#[derive(Debug, Clone)]
pub struct TfIdfEmbedder {
    dimension: usize,
    vocabulary: std::collections::HashMap<String, usize>,
    idf: Vec<f32>,
}

impl TfIdfEmbedder {
    /// Create a new TF-IDF embedder (untrained)
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vocabulary: std::collections::HashMap::new(),
            idf: Vec::new(),
        }
    }

    /// Train the embedder on a corpus
    pub fn fit(&mut self, documents: &[&str]) {
        use std::collections::{HashMap, HashSet};

        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut all_terms: HashSet<String> = HashSet::new();

        for doc in documents {
            let terms: HashSet<String> = doc.split_whitespace().map(|s| s.to_lowercase()).collect();

            for term in &terms {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
                all_terms.insert(term.clone());
            }
        }

        // Build vocabulary (top N terms by document frequency)
        let mut terms: Vec<_> = all_terms.into_iter().collect();
        terms.sort_by_key(|t| std::cmp::Reverse(doc_freq.get(t).copied().unwrap_or(0)));
        terms.truncate(self.dimension);

        self.vocabulary = terms
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        // Compute IDF
        let n = documents.len() as f32;
        self.idf = terms
            .iter()
            .map(|t| {
                let df = doc_freq.get(t).copied().unwrap_or(1) as f32;
                (n / df).ln() + 1.0
            })
            .collect();
    }

    fn compute_tf(&self, text: &str) -> Vec<f32> {
        let mut tf = vec![0.0f32; self.dimension];
        let terms: Vec<String> = text.split_whitespace().map(|s| s.to_lowercase()).collect();
        let total = terms.len() as f32;

        for term in terms {
            if let Some(&idx) = self.vocabulary.get(&term) {
                tf[idx] += 1.0 / total;
            }
        }

        tf
    }
}

impl Embedder for TfIdfEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::EmptyDocument("empty text".to_string()));
        }

        if self.vocabulary.is_empty() {
            return Err(Error::InvalidConfig("embedder not trained".to_string()));
        }

        let tf = self.compute_tf(text);
        let mut tfidf: Vec<f32> = tf.iter().zip(self.idf.iter()).map(|(t, i)| t * i).collect();

        // Normalize
        let norm: f32 = tfidf.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut tfidf {
                *x /= norm;
            }
        }

        // Pad to dimension if needed
        tfidf.resize(self.dimension, 0.0);
        Ok(tfidf)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        "tfidf"
    }
}

/// Compute cosine similarity between two vectors
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Compute dot product between two vectors
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute euclidean distance between two vectors
#[must_use]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ PoolingStrategy Tests ============

    #[test]
    fn test_pooling_strategy_default() {
        assert_eq!(PoolingStrategy::default(), PoolingStrategy::Mean);
    }

    #[test]
    fn test_pooling_strategy_serialization() {
        let strategy = PoolingStrategy::WeightedMean;
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: PoolingStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(strategy, deserialized);
    }

    // ============ EmbeddingConfig Tests ============

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert!(config.normalize);
        assert!(config.query_prefix.is_none());
        assert!(config.document_prefix.is_none());
        assert_eq!(config.max_length, 512);
        assert_eq!(config.pooling, PoolingStrategy::Mean);
    }

    #[test]
    fn test_embedding_config_serialization() {
        let config = EmbeddingConfig {
            normalize: false,
            query_prefix: Some("query: ".to_string()),
            document_prefix: Some("passage: ".to_string()),
            max_length: 256,
            pooling: PoolingStrategy::Cls,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EmbeddingConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.normalize, deserialized.normalize);
        assert_eq!(config.query_prefix, deserialized.query_prefix);
        assert_eq!(config.max_length, deserialized.max_length);
    }

    // ============ MockEmbedder Tests ============

    #[test]
    fn test_mock_embedder_new() {
        let embedder = MockEmbedder::new(384);
        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.model_id(), "mock-embedder");
    }

    #[test]
    fn test_mock_embedder_with_model_id() {
        let embedder = MockEmbedder::new(768).with_model_id("custom-model");
        assert_eq!(embedder.model_id(), "custom-model");
    }

    #[test]
    fn test_mock_embedder_embed() {
        let embedder = MockEmbedder::new(128);
        let embedding = embedder.embed("Hello world").unwrap();

        assert_eq!(embedding.len(), 128);
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mock_embedder_embed_empty() {
        let embedder = MockEmbedder::new(128);
        let result = embedder.embed("");
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_embedder_embed_batch() {
        let embedder = MockEmbedder::new(64);
        let texts = vec!["Hello", "World", "Test"];
        let embeddings = embedder.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 64);
        }
    }

    #[test]
    fn test_mock_embedder_deterministic() {
        let embedder = MockEmbedder::new(128);
        let emb1 = embedder.embed("Hello").unwrap();
        let emb2 = embedder.embed("Hello").unwrap();
        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_mock_embedder_different_texts() {
        let embedder = MockEmbedder::new(128);
        let emb1 = embedder.embed("Hello").unwrap();
        let emb2 = embedder.embed("World").unwrap();
        assert_ne!(emb1, emb2);
    }

    #[test]
    fn test_mock_embedder_query_prefix() {
        let config = EmbeddingConfig {
            query_prefix: Some("query: ".to_string()),
            ..Default::default()
        };
        let embedder = MockEmbedder::new(128).with_config(config);

        let query_emb = embedder.embed_query("test").unwrap();
        let doc_emb = embedder.embed_document("test").unwrap();

        // Query and doc embeddings should differ due to prefix
        assert_ne!(query_emb, doc_emb);
    }

    #[test]
    fn test_mock_embedder_embed_chunks() {
        use crate::DocumentId;
        let embedder = MockEmbedder::new(64);

        let doc_id = DocumentId::new();
        let mut chunks = vec![
            Chunk::new(doc_id, "First chunk".to_string(), 0, 11),
            Chunk::new(doc_id, "Second chunk".to_string(), 12, 24),
        ];

        embedder.embed_chunks(&mut chunks).unwrap();

        for chunk in &chunks {
            assert!(chunk.embedding.is_some());
            assert_eq!(chunk.embedding.as_ref().unwrap().len(), 64);
        }
    }

    // ============ TfIdfEmbedder Tests ============

    #[test]
    fn test_tfidf_embedder_new() {
        let embedder = TfIdfEmbedder::new(100);
        assert_eq!(embedder.dimension(), 100);
        assert_eq!(embedder.model_id(), "tfidf");
    }

    #[test]
    fn test_tfidf_embedder_untrained() {
        let embedder = TfIdfEmbedder::new(100);
        let result = embedder.embed("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_tfidf_embedder_fit() {
        let mut embedder = TfIdfEmbedder::new(50);
        let corpus = vec!["the quick brown fox", "the lazy dog", "quick brown dog"];
        embedder.fit(&corpus);

        assert!(!embedder.vocabulary.is_empty());
        assert!(!embedder.idf.is_empty());
    }

    #[test]
    fn test_tfidf_embedder_embed() {
        let mut embedder = TfIdfEmbedder::new(50);
        let corpus = vec![
            "the quick brown fox",
            "the lazy dog sleeps",
            "quick brown lazy fox",
        ];
        embedder.fit(&corpus);

        let embedding = embedder.embed("quick fox").unwrap();
        assert_eq!(embedding.len(), 50);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
    }

    #[test]
    fn test_tfidf_embedder_empty() {
        let mut embedder = TfIdfEmbedder::new(50);
        embedder.fit(&["test doc"]);
        let result = embedder.embed("");
        assert!(result.is_err());
    }

    // ============ Similarity Functions Tests ============

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b);
        assert!((dot - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dot = dot_product(&a, &b);
        assert!(dot.abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance_same() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = euclidean_distance(&a, &b);
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_mock_embedder_dimension(dim in 1usize..1000) {
            let embedder = MockEmbedder::new(dim);
            let emb = embedder.embed("test").unwrap();
            prop_assert_eq!(emb.len(), dim);
        }

        #[test]
        fn prop_mock_embedder_normalized(text in "[a-zA-Z ]{1,100}") {
            let embedder = MockEmbedder::new(128);
            let emb = embedder.embed(&text).unwrap();
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!((norm - 1.0).abs() < 1e-4);
        }

        #[test]
        fn prop_cosine_similarity_range(
            a in prop::collection::vec(-1.0f32..1.0, 10),
            b in prop::collection::vec(-1.0f32..1.0, 10)
        ) {
            let sim = cosine_similarity(&a, &b);
            prop_assert!(sim >= -1.0 - 1e-5);
            prop_assert!(sim <= 1.0 + 1e-5);
        }

        #[test]
        fn prop_euclidean_distance_non_negative(
            a in prop::collection::vec(-100.0f32..100.0, 5),
            b in prop::collection::vec(-100.0f32..100.0, 5)
        ) {
            let dist = euclidean_distance(&a, &b);
            prop_assert!(dist >= 0.0);
        }
    }
}
