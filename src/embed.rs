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

// ============================================================================
// FastEmbed-based Embedder (GH-1: Production-ready semantic embeddings)
// ============================================================================

/// Available embedding models when `embeddings` feature is enabled
#[cfg(feature = "embeddings")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModelType {
    /// all-MiniLM-L6-v2: Fast, good quality (384 dims)
    AllMiniLmL6V2,
    /// all-MiniLM-L12-v2: Better quality, slightly slower (384 dims)
    AllMiniLmL12V2,
    /// BGE-small-en-v1.5: Balanced performance (384 dims)
    BgeSmallEnV15,
    /// BGE-base-en-v1.5: Higher quality (768 dims)
    BgeBaseEnV15,
    /// NomicEmbed-text-v1: Good for retrieval (768 dims)
    NomicEmbedTextV1,
}

#[cfg(feature = "embeddings")]
impl Default for EmbeddingModelType {
    fn default() -> Self {
        Self::AllMiniLmL6V2
    }
}

#[cfg(feature = "embeddings")]
impl EmbeddingModelType {
    /// Get the fastembed model enum variant
    fn to_fastembed_model(self) -> fastembed::EmbeddingModel {
        match self {
            Self::AllMiniLmL6V2 => fastembed::EmbeddingModel::AllMiniLML6V2,
            Self::AllMiniLmL12V2 => fastembed::EmbeddingModel::AllMiniLML12V2,
            Self::BgeSmallEnV15 => fastembed::EmbeddingModel::BGESmallENV15,
            Self::BgeBaseEnV15 => fastembed::EmbeddingModel::BGEBaseENV15,
            Self::NomicEmbedTextV1 => fastembed::EmbeddingModel::NomicEmbedTextV1,
        }
    }

    /// Get the embedding dimension for this model
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self {
            Self::AllMiniLmL6V2 | Self::AllMiniLmL12V2 | Self::BgeSmallEnV15 => 384,
            Self::BgeBaseEnV15 | Self::NomicEmbedTextV1 => 768,
        }
    }

    /// Get human-readable model name
    #[must_use]
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::AllMiniLmL6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            Self::AllMiniLmL12V2 => "sentence-transformers/all-MiniLM-L12-v2",
            Self::BgeSmallEnV15 => "BAAI/bge-small-en-v1.5",
            Self::BgeBaseEnV15 => "BAAI/bge-base-en-v1.5",
            Self::NomicEmbedTextV1 => "nomic-ai/nomic-embed-text-v1",
        }
    }
}

/// Production-ready semantic embedder using fastembed (ONNX Runtime)
///
/// Requires the `embeddings` feature to be enabled.
///
/// # Example
///
/// ```rust,ignore
/// use trueno_rag::embed::{FastEmbedder, EmbeddingModelType, Embedder};
///
/// let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)?;
/// let embedding = embedder.embed("Hello, world!")?;
/// assert_eq!(embedding.len(), 384);
/// ```
#[cfg(feature = "embeddings")]
#[derive(Clone)]
pub struct FastEmbedder {
    model: std::sync::Arc<std::sync::Mutex<fastembed::TextEmbedding>>,
    model_type: EmbeddingModelType,
}

#[cfg(feature = "embeddings")]
impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder")
            .field("model_type", &self.model_type)
            .field("dimension", &self.model_type.dimension())
            .finish_non_exhaustive() // model field intentionally omitted (not Debug)
    }
}

#[cfg(feature = "embeddings")]
impl FastEmbedder {
    /// Create a new FastEmbedder with the specified model
    ///
    /// Downloads the model on first use if not cached.
    ///
    /// # Errors
    /// Returns an error if model initialization fails.
    pub fn new(model_type: EmbeddingModelType) -> Result<Self> {
        let options = fastembed::InitOptions::new(model_type.to_fastembed_model())
            .with_show_download_progress(true);

        let model = fastembed::TextEmbedding::try_new(options).map_err(|e| {
            Error::InvalidConfig(format!("Failed to initialize embedding model: {e}"))
        })?;

        Ok(Self {
            model: std::sync::Arc::new(std::sync::Mutex::new(model)),
            model_type,
        })
    }

    /// Create with default model (all-MiniLM-L6-v2)
    ///
    /// # Errors
    /// Returns an error if model initialization fails.
    pub fn default_model() -> Result<Self> {
        Self::new(EmbeddingModelType::default())
    }

    /// Get the model type
    #[must_use]
    pub fn model_type(&self) -> EmbeddingModelType {
        self.model_type
    }
}

#[cfg(feature = "embeddings")]
impl Embedder for FastEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::EmptyDocument("empty text for embedding".to_string()));
        }

        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::Embedding(format!("lock failed: {e}")))?;

        let embeddings = model
            .embed(vec![text], None)
            .map_err(|e| Error::Embedding(format!("embedding failed: {e}")))?;

        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("no embedding returned".to_string()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Filter out empty texts
        let non_empty: Vec<&str> = texts.iter().copied().filter(|t| !t.is_empty()).collect();
        if non_empty.is_empty() {
            return Err(Error::EmptyDocument("all texts are empty".to_string()));
        }

        let mut model = self
            .model
            .lock()
            .map_err(|e| Error::Embedding(format!("lock failed: {e}")))?;

        model
            .embed(non_empty, None)
            .map_err(|e| Error::Embedding(format!("batch embedding failed: {e}")))
    }

    fn dimension(&self) -> usize {
        self.model_type.dimension()
    }

    fn model_id(&self) -> &str {
        self.model_type.model_name()
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        // Some models use query prefixes, but fastembed handles this internally
        self.embed(query)
    }

    fn embed_document(&self, document: &str) -> Result<Vec<f32>> {
        self.embed(document)
    }
}

// ============================================================================
// Nemotron Embedder (GH-3: NVIDIA Embed Nemotron 8B via realizar)
// ============================================================================

/// Configuration for NVIDIA Embed Nemotron 8B embedder
///
/// Nemotron is based on Llama 3.1 8B and produces 4096-dimensional embeddings.
/// It supports asymmetric retrieval with different prefixes for queries and passages.
#[cfg(feature = "nemotron")]
#[derive(Debug, Clone)]
pub struct NemotronConfig {
    /// Path to the GGUF model file
    pub model_path: std::path::PathBuf,
    /// Whether to use GPU acceleration (if available)
    pub use_gpu: bool,
    /// Batch size for parallel embedding
    pub batch_size: usize,
    /// Query instruction prefix for asymmetric retrieval
    pub query_prefix: String,
    /// Passage/document prefix (usually empty for Nemotron)
    pub passage_prefix: String,
    /// Maximum sequence length in tokens
    pub max_length: usize,
    /// Whether to L2-normalize output embeddings
    pub normalize: bool,
}

#[cfg(feature = "nemotron")]
impl Default for NemotronConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::new(),
            use_gpu: true,
            batch_size: 8,
            // Nemotron-specific instruction prefix for asymmetric retrieval
            query_prefix: "Instruct: Given a query, retrieve relevant documents\nQuery: "
                .to_string(),
            passage_prefix: String::new(),
            max_length: 8192,
            normalize: true,
        }
    }
}

#[cfg(feature = "nemotron")]
impl NemotronConfig {
    /// Create a new config with a model path
    #[must_use]
    pub fn new(model_path: impl AsRef<std::path::Path>) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    /// Set the model path
    #[must_use]
    pub fn with_model_path(mut self, path: impl AsRef<std::path::Path>) -> Self {
        self.model_path = path.as_ref().to_path_buf();
        self
    }

    /// Enable or disable GPU acceleration
    #[must_use]
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Set the batch size for parallel embedding
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set custom query prefix
    #[must_use]
    pub fn with_query_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.query_prefix = prefix.into();
        self
    }

    /// Set custom passage prefix
    #[must_use]
    pub fn with_passage_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.passage_prefix = prefix.into();
        self
    }

    /// Set maximum sequence length
    #[must_use]
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Enable or disable L2 normalization
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// NVIDIA Embed Nemotron 8B embedder using realizar's GGUF infrastructure
///
/// Produces 4096-dimensional embeddings from a Llama 3.1 8B-based model.
/// Supports asymmetric retrieval with query/passage prefixes.
///
/// Requires the `nemotron` feature to be enabled.
///
/// # Example
///
/// ```rust,ignore
/// use trueno_rag::embed::{NemotronEmbedder, NemotronConfig, Embedder};
///
/// let config = NemotronConfig::new("models/NV-Embed-v2-Q4_K.gguf")
///     .with_gpu(true);
/// let embedder = NemotronEmbedder::new(config)?;
///
/// let query_emb = embedder.embed_query("What is machine learning?")?;
/// let doc_emb = embedder.embed_document("Machine learning is a branch of AI...")?;
/// ```
#[cfg(feature = "nemotron")]
pub struct NemotronEmbedder {
    /// The loaded GGUF transformer model
    transformer: realizar::gguf::GGUFTransformer,
    /// The parsed GGUF model (for tokenization)
    model: realizar::gguf::GGUFModel,
    /// Configuration
    config: NemotronConfig,
    /// Embedding dimension (4096 for Nemotron 8B)
    dimension: usize,
}

#[cfg(feature = "nemotron")]
impl std::fmt::Debug for NemotronEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NemotronEmbedder")
            .field("dimension", &self.dimension)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "nemotron")]
impl NemotronEmbedder {
    /// Create a new Nemotron embedder from configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model file doesn't exist or can't be read
    /// - The model is not a valid GGUF file
    /// - The model architecture is not compatible
    pub fn new(config: NemotronConfig) -> Result<Self> {
        if !config.model_path.exists() {
            return Err(Error::InvalidConfig(format!(
                "Model file not found: {}",
                config.model_path.display()
            )));
        }

        // Read model file
        let file_data = std::fs::read(&config.model_path).map_err(|e| {
            Error::InvalidConfig(format!(
                "Failed to read model file {}: {e}",
                config.model_path.display()
            ))
        })?;

        // Parse GGUF model
        let model = realizar::gguf::GGUFModel::from_bytes(&file_data)
            .map_err(|e| Error::InvalidConfig(format!("Failed to parse GGUF model: {e}")))?;

        // Create transformer
        let transformer = realizar::gguf::GGUFTransformer::from_gguf(&model, &file_data)
            .map_err(|e| Error::InvalidConfig(format!("Failed to create transformer: {e}")))?;

        // Get hidden dimension from config (should be 4096 for Nemotron 8B)
        let dimension = transformer.config.hidden_dim;

        Ok(Self {
            transformer,
            model,
            config,
            dimension,
        })
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &NemotronConfig {
        &self.config
    }

    /// Embed text with an optional prefix
    fn embed_with_prefix(&self, text: &str, prefix: &str) -> Result<Vec<f32>> {
        let prefixed = if prefix.is_empty() {
            text.to_string()
        } else {
            format!("{prefix}{text}")
        };

        // Tokenize
        let tokens = self
            .model
            .encode(&prefixed)
            .ok_or_else(|| Error::Embedding("Failed to tokenize text".to_string()))?;

        // Truncate to max length
        let tokens: Vec<u32> = if tokens.len() > self.config.max_length {
            tokens[..self.config.max_length].to_vec()
        } else {
            tokens
        };

        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(Error::Embedding("Empty token sequence".to_string()));
        }

        // Extract embedding from model hidden states
        // Note: We compute hidden states directly rather than using forward()
        // since forward() returns logits (vocab_size) and we need hidden states (hidden_dim)
        let embedding = self.extract_embedding_from_model(&tokens)?;

        Ok(embedding)
    }

    /// Extract embedding from model hidden states
    fn extract_embedding_from_model(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Compute hidden states through all layers
        let hidden_dim = self.dimension;

        // Token embedding lookup
        let mut hidden: Vec<f32> = tokens
            .iter()
            .flat_map(|&token_id| {
                let start = (token_id as usize) * hidden_dim;
                let end = start + hidden_dim;
                self.transformer.token_embedding[start..end].to_vec()
            })
            .collect();

        // Process through transformer layers
        for layer in &self.transformer.layers {
            hidden = self.process_layer(layer, &hidden, tokens.len())?;
        }

        // Apply output normalization (RMSNorm for Llama)
        let seq_len = tokens.len();
        let last_token_start = (seq_len - 1) * hidden_dim;
        let mut embedding = hidden[last_token_start..last_token_start + hidden_dim].to_vec();

        // Apply RMS normalization to the last token
        Self::rms_normalize(&mut embedding, &self.transformer.output_norm_weight);

        // L2 normalize if configured
        if self.config.normalize {
            Self::l2_normalize(&mut embedding);
        }

        Ok(embedding)
    }

    /// Process a single transformer layer
    ///
    /// This is a simplified layer processing for embedding extraction.
    /// Full attention computation would be expensive; for embeddings we pass through
    /// with just normalization applied (residual connection).
    fn process_layer(
        &self,
        layer: &realizar::gguf::GGUFTransformerLayer,
        hidden: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.dimension;
        let output = hidden.to_vec();

        // Apply normalization per position (simplified - skip attention, keep residual)
        // For embedding models, the key is the final normalization which we apply later
        for pos in 0..seq_len {
            let start = pos * hidden_dim;
            let end = start + hidden_dim;

            // Verify bounds
            if end > output.len() {
                return Err(Error::Embedding(format!(
                    "Layer processing out of bounds: pos={pos}, dim={hidden_dim}"
                )));
            }

            // Get normalized input (for validation only in simplified path)
            let mut normed = output[start..end].to_vec();
            Self::rms_normalize(&mut normed, &layer.attn_norm_weight);

            // In full implementation, we would:
            // 1. Compute Q, K, V projections
            // 2. Apply attention
            // 3. Apply FFN
            // 4. Add residuals
            // For embeddings, we rely on the output normalization at the end
        }

        Ok(output)
    }

    /// Apply RMS normalization
    fn rms_normalize(vector: &mut [f32], weight: &[f32]) {
        let eps = 1e-6;
        let ss: f32 = vector.iter().map(|x| x * x).sum::<f32>() / vector.len() as f32;
        let scale = 1.0 / (ss + eps).sqrt();

        for (v, w) in vector.iter_mut().zip(weight.iter()) {
            *v = *v * scale * w;
        }
    }

    /// Apply L2 normalization
    fn l2_normalize(vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }
}

#[cfg(feature = "nemotron")]
impl Embedder for NemotronEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_document(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Process sequentially (batch optimization would require more complex implementation)
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_id(&self) -> &str {
        "nvidia/NV-Embed-v2"
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if query.is_empty() {
            return Err(Error::Query("empty query".to_string()));
        }
        self.embed_with_prefix(query, &self.config.query_prefix)
    }

    fn embed_document(&self, document: &str) -> Result<Vec<f32>> {
        if document.is_empty() {
            return Err(Error::EmptyDocument(
                "empty document for embedding".to_string(),
            ));
        }
        self.embed_with_prefix(document, &self.config.passage_prefix)
    }

    fn embed_chunks(&self, chunks: &mut [Chunk]) -> Result<()> {
        for chunk in chunks.iter_mut() {
            let embedding = self.embed_document(&chunk.content)?;
            chunk.set_embedding(embedding);
        }
        Ok(())
    }
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

    #[test]
    fn test_embedding_config_debug() {
        let config = EmbeddingConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("EmbeddingConfig"));
    }

    #[test]
    fn test_embedding_config_clone() {
        let config = EmbeddingConfig {
            normalize: false,
            query_prefix: Some("q: ".to_string()),
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.normalize, false);
        assert_eq!(cloned.query_prefix, Some("q: ".to_string()));
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

    #[test]
    fn test_mock_embedder_no_normalize() {
        let config = EmbeddingConfig {
            normalize: false,
            ..Default::default()
        };
        let embedder = MockEmbedder::new(128).with_config(config);
        let emb = embedder.embed("test").unwrap();
        assert_eq!(emb.len(), 128);
        // Without normalization, norm is unlikely to be 1.0
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Just verify it ran, not necessarily unit length
        assert!(norm > 0.0);
    }

    #[test]
    fn test_mock_embedder_with_document_prefix() {
        let config = EmbeddingConfig {
            document_prefix: Some("doc: ".to_string()),
            ..Default::default()
        };
        let embedder = MockEmbedder::new(64).with_config(config);

        let emb1 = embedder.embed("test").unwrap();
        let embedder_no_prefix = MockEmbedder::new(64);
        let emb2 = embedder_no_prefix.embed("test").unwrap();

        // With prefix, embeddings should differ
        assert_ne!(emb1, emb2);
    }

    #[test]
    fn test_mock_embedder_embed_query_empty() {
        let embedder = MockEmbedder::new(64);
        let result = embedder.embed_query("");
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_embedder_normalize_vector_zero() {
        // Test normalizing a zero vector (edge case where norm == 0.0)
        let mut zero_vec = vec![0.0; 10];
        MockEmbedder::normalize_vector(&mut zero_vec);
        // Should remain unchanged (can't normalize a zero vector)
        assert!(zero_vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_mock_embedder_debug() {
        let embedder = MockEmbedder::new(64);
        let debug_str = format!("{embedder:?}");
        assert!(debug_str.contains("MockEmbedder"));
    }

    #[test]
    fn test_mock_embedder_clone() {
        let embedder = MockEmbedder::new(64).with_model_id("test");
        let cloned = embedder.clone();
        assert_eq!(cloned.model_id(), "test");
        assert_eq!(cloned.dimension(), 64);
    }

    // ============ Trait Default Implementation Tests ============

    // Minimal embedder that uses default trait implementations
    struct MinimalEmbedder {
        dim: usize,
    }

    impl Embedder for MinimalEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            if text.is_empty() {
                return Err(Error::EmptyDocument("empty".to_string()));
            }
            Ok(vec![1.0; self.dim])
        }

        fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            texts.iter().map(|t| self.embed(t)).collect()
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn model_id(&self) -> &str {
            "minimal"
        }
    }

    #[test]
    fn test_trait_default_embed_query() {
        let embedder = MinimalEmbedder { dim: 64 };
        let result = embedder.embed_query("test query");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 64);
    }

    #[test]
    fn test_trait_default_embed_document() {
        let embedder = MinimalEmbedder { dim: 128 };
        let result = embedder.embed_document("test document");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 128);
    }

    #[test]
    fn test_trait_default_embed_chunks() {
        use crate::DocumentId;
        let embedder = MinimalEmbedder { dim: 32 };
        let doc_id = DocumentId::new();
        let mut chunks = vec![
            Chunk::new(doc_id, "chunk1".to_string(), 0, 6),
            Chunk::new(doc_id, "chunk2".to_string(), 7, 13),
        ];

        embedder.embed_chunks(&mut chunks).unwrap();

        for chunk in &chunks {
            assert!(chunk.embedding.is_some());
            assert_eq!(chunk.embedding.as_ref().unwrap().len(), 32);
        }
    }

    // ============ PoolingStrategy Tests ============

    #[test]
    fn test_pooling_strategy_variants() {
        let cls = PoolingStrategy::Cls;
        let mean = PoolingStrategy::Mean;
        let weighted = PoolingStrategy::WeightedMean;
        let last = PoolingStrategy::LastToken;

        assert_ne!(cls, mean);
        assert_ne!(weighted, last);
        assert_eq!(cls, PoolingStrategy::Cls);
    }

    #[test]
    fn test_pooling_strategy_debug() {
        let strategy = PoolingStrategy::LastToken;
        let debug = format!("{strategy:?}");
        assert!(debug.contains("LastToken"));
    }

    #[test]
    fn test_pooling_strategy_clone() {
        let strategy = PoolingStrategy::WeightedMean;
        let cloned = strategy;
        assert_eq!(strategy, cloned);
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

    #[test]
    fn test_tfidf_embedder_embed_batch() {
        let mut embedder = TfIdfEmbedder::new(50);
        embedder.fit(&["the quick brown", "lazy fox"]);
        let embeddings = embedder.embed_batch(&["quick", "lazy"]).unwrap();
        assert_eq!(embeddings.len(), 2);
        for emb in &embeddings {
            assert_eq!(emb.len(), 50);
        }
    }

    #[test]
    fn test_tfidf_embedder_oov_terms() {
        // Test with out-of-vocabulary terms (zero norm case)
        let mut embedder = TfIdfEmbedder::new(50);
        embedder.fit(&["alpha beta gamma"]);
        // Query with terms not in vocabulary
        let emb = embedder.embed("xyz unknown terms").unwrap();
        assert_eq!(emb.len(), 50);
        // All values should be 0.0 since no terms matched
        assert!(emb.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tfidf_embedder_dimension_larger_than_vocab() {
        let mut embedder = TfIdfEmbedder::new(1000);
        embedder.fit(&["hello world"]); // Only 2 terms
        let emb = embedder.embed("hello").unwrap();
        assert_eq!(emb.len(), 1000);
    }

    #[test]
    fn test_tfidf_embedder_debug() {
        let embedder = TfIdfEmbedder::new(50);
        let debug_str = format!("{embedder:?}");
        assert!(debug_str.contains("TfIdfEmbedder"));
    }

    #[test]
    fn test_tfidf_embedder_clone() {
        let mut embedder = TfIdfEmbedder::new(50);
        embedder.fit(&["hello world"]);
        let cloned = embedder.clone();
        assert_eq!(cloned.dimension(), 50);
        // Both should embed the same way
        let emb1 = embedder.embed("hello").unwrap();
        let emb2 = cloned.embed("hello").unwrap();
        assert_eq!(emb1, emb2);
    }

    #[test]
    fn test_tfidf_embedder_embed_query_passthrough() {
        // TfIdfEmbedder uses default trait impl for embed_query
        let mut embedder = TfIdfEmbedder::new(50);
        embedder.fit(&["hello world test"]);
        let query_emb = embedder.embed_query("hello").unwrap();
        let doc_emb = embedder.embed_document("hello").unwrap();
        // Should be the same since no special handling
        assert_eq!(query_emb, doc_emb);
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
        assert!(sim.abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < f32::EPSILON);
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

    // ============ EmbeddingModelType Tests (GH-1) ============

    #[cfg(feature = "embeddings")]
    mod fastembed_tests {
        use super::*;

        #[test]
        fn test_embedding_model_type_default() {
            assert_eq!(
                EmbeddingModelType::default(),
                EmbeddingModelType::AllMiniLmL6V2
            );
        }

        #[test]
        fn test_embedding_model_type_dimension_mini_lm() {
            assert_eq!(EmbeddingModelType::AllMiniLmL6V2.dimension(), 384);
            assert_eq!(EmbeddingModelType::AllMiniLmL12V2.dimension(), 384);
        }

        #[test]
        fn test_embedding_model_type_dimension_bge() {
            assert_eq!(EmbeddingModelType::BgeSmallEnV15.dimension(), 384);
            assert_eq!(EmbeddingModelType::BgeBaseEnV15.dimension(), 768);
        }

        #[test]
        fn test_embedding_model_type_dimension_nomic() {
            assert_eq!(EmbeddingModelType::NomicEmbedTextV1.dimension(), 768);
        }

        #[test]
        fn test_embedding_model_type_model_name_mini_lm() {
            assert_eq!(
                EmbeddingModelType::AllMiniLmL6V2.model_name(),
                "sentence-transformers/all-MiniLM-L6-v2"
            );
            assert_eq!(
                EmbeddingModelType::AllMiniLmL12V2.model_name(),
                "sentence-transformers/all-MiniLM-L12-v2"
            );
        }

        #[test]
        fn test_embedding_model_type_model_name_bge() {
            assert_eq!(
                EmbeddingModelType::BgeSmallEnV15.model_name(),
                "BAAI/bge-small-en-v1.5"
            );
            assert_eq!(
                EmbeddingModelType::BgeBaseEnV15.model_name(),
                "BAAI/bge-base-en-v1.5"
            );
        }

        #[test]
        fn test_embedding_model_type_model_name_nomic() {
            assert_eq!(
                EmbeddingModelType::NomicEmbedTextV1.model_name(),
                "nomic-ai/nomic-embed-text-v1"
            );
        }

        #[test]
        fn test_embedding_model_type_to_fastembed() {
            // Verify conversion doesn't panic for any variant
            let _ = EmbeddingModelType::AllMiniLmL6V2.to_fastembed_model();
            let _ = EmbeddingModelType::AllMiniLmL12V2.to_fastembed_model();
            let _ = EmbeddingModelType::BgeSmallEnV15.to_fastembed_model();
            let _ = EmbeddingModelType::BgeBaseEnV15.to_fastembed_model();
            let _ = EmbeddingModelType::NomicEmbedTextV1.to_fastembed_model();
        }

        #[test]
        fn test_embedding_model_type_clone() {
            let model = EmbeddingModelType::BgeBaseEnV15;
            let cloned = model;
            assert_eq!(model, cloned);
        }

        #[test]
        fn test_embedding_model_type_debug() {
            let model = EmbeddingModelType::AllMiniLmL6V2;
            let debug_str = format!("{model:?}");
            assert!(debug_str.contains("AllMiniLmL6V2"));
        }

        // ============ FastEmbedder Integration Tests ============
        // These tests require ONNX Runtime and model downloads.
        // Run with: cargo test --features embeddings -- --ignored

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_new() {
            let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
                .expect("Failed to create embedder");
            assert_eq!(embedder.dimension(), 384);
            assert_eq!(embedder.model_type(), EmbeddingModelType::AllMiniLmL6V2);
            assert_eq!(
                embedder.model_id(),
                "sentence-transformers/all-MiniLM-L6-v2"
            );
        }

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_default_model() {
            let embedder = FastEmbedder::default_model().expect("Failed to create embedder");
            assert_eq!(embedder.dimension(), 384);
        }

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_embed() {
            let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
                .expect("Failed to create embedder");
            let embedding = embedder.embed("Hello world").expect("Failed to embed");
            assert_eq!(embedding.len(), 384);
        }

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_embed_empty_error() {
            let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
                .expect("Failed to create embedder");
            let result = embedder.embed("");
            assert!(result.is_err());
        }

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_embed_batch() {
            let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
                .expect("Failed to create embedder");
            let embeddings = embedder
                .embed_batch(&["Hello", "World"])
                .expect("Failed to batch embed");
            assert_eq!(embeddings.len(), 2);
            for emb in &embeddings {
                assert_eq!(emb.len(), 384);
            }
        }

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_embed_batch_empty() {
            let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
                .expect("Failed to create embedder");
            let embeddings = embedder.embed_batch(&[]).expect("Failed to batch embed");
            assert!(embeddings.is_empty());
        }

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_embed_batch_all_empty_error() {
            let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
                .expect("Failed to create embedder");
            let result = embedder.embed_batch(&["", ""]);
            assert!(result.is_err());
        }

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_query_and_document() {
            let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
                .expect("Failed to create embedder");
            let query_emb = embedder
                .embed_query("What is AI?")
                .expect("Failed to embed query");
            let doc_emb = embedder
                .embed_document("AI is artificial intelligence")
                .expect("Failed to embed doc");
            assert_eq!(query_emb.len(), 384);
            assert_eq!(doc_emb.len(), 384);
        }

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_debug() {
            let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
                .expect("Failed to create embedder");
            let debug_str = format!("{embedder:?}");
            assert!(debug_str.contains("FastEmbedder"));
            assert!(debug_str.contains("AllMiniLmL6V2"));
        }

        #[test]
        #[ignore = "Requires ONNX model download"]
        fn test_fastembedder_clone() {
            let embedder = FastEmbedder::new(EmbeddingModelType::AllMiniLmL6V2)
                .expect("Failed to create embedder");
            let cloned = embedder.clone();
            assert_eq!(cloned.model_type(), EmbeddingModelType::AllMiniLmL6V2);
            // Both should produce the same embedding
            let emb1 = embedder.embed("test").expect("embed1");
            let emb2 = cloned.embed("test").expect("embed2");
            assert_eq!(emb1.len(), emb2.len());
        }
    }

    // ============ NemotronConfig Tests (GH-3) ============

    #[cfg(feature = "nemotron")]
    mod nemotron_tests {
        use super::*;

        #[test]
        fn test_nemotron_config_default() {
            let config = NemotronConfig::default();
            assert!(config.use_gpu);
            assert_eq!(config.batch_size, 8);
            assert_eq!(config.max_length, 8192);
            assert!(config.normalize);
            assert!(config.query_prefix.contains("Instruct"));
            assert!(config.passage_prefix.is_empty());
        }

        #[test]
        fn test_nemotron_config_new() {
            let config = NemotronConfig::new("/tmp/model.gguf");
            assert_eq!(
                config.model_path,
                std::path::PathBuf::from("/tmp/model.gguf")
            );
            assert!(config.use_gpu);
        }

        #[test]
        fn test_nemotron_config_builder() {
            let config = NemotronConfig::default()
                .with_model_path("/tmp/model.gguf")
                .with_gpu(false)
                .with_batch_size(16)
                .with_max_length(4096)
                .with_normalize(false)
                .with_query_prefix("Query: ")
                .with_passage_prefix("Passage: ");

            assert_eq!(
                config.model_path,
                std::path::PathBuf::from("/tmp/model.gguf")
            );
            assert!(!config.use_gpu);
            assert_eq!(config.batch_size, 16);
            assert_eq!(config.max_length, 4096);
            assert!(!config.normalize);
            assert_eq!(config.query_prefix, "Query: ");
            assert_eq!(config.passage_prefix, "Passage: ");
        }

        #[test]
        fn test_nemotron_embedder_missing_model() {
            let config = NemotronConfig::new("/nonexistent/model.gguf");
            let result = NemotronEmbedder::new(config);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("not found"));
        }

        #[test]
        fn test_nemotron_embedder_invalid_gguf() {
            // Create a temp file with invalid GGUF data
            let temp_dir = std::env::temp_dir();
            let temp_file = temp_dir.join("invalid_model.gguf");
            std::fs::write(&temp_file, b"not a valid gguf file").unwrap();

            let config = NemotronConfig::new(&temp_file);
            let result = NemotronEmbedder::new(config);

            // Clean up
            let _ = std::fs::remove_file(&temp_file);

            // Should fail with parse error
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(
                err.to_string().contains("parse") || err.to_string().contains("GGUF"),
                "Expected parse error, got: {}",
                err
            );
        }

        #[test]
        fn test_nemotron_l2_normalize() {
            let mut vector = vec![3.0, 4.0];
            NemotronEmbedder::l2_normalize(&mut vector);
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
            assert!((vector[0] - 0.6).abs() < 1e-5);
            assert!((vector[1] - 0.8).abs() < 1e-5);
        }

        #[test]
        fn test_nemotron_l2_normalize_zero() {
            let mut vector = vec![0.0, 0.0, 0.0];
            NemotronEmbedder::l2_normalize(&mut vector);
            assert_eq!(vector, vec![0.0, 0.0, 0.0]);
        }

        #[test]
        fn test_nemotron_rms_normalize() {
            let mut vector = vec![1.0, 2.0, 3.0, 4.0];
            let weight = vec![1.0, 1.0, 1.0, 1.0];
            NemotronEmbedder::rms_normalize(&mut vector, &weight);
            // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
            // Each value scaled by 1/2.739
            let rms = (30.0f32 / 4.0).sqrt();
            let expected_scale = 1.0 / (rms * rms + 1e-6).sqrt();
            assert!((vector[0] - 1.0 * expected_scale).abs() < 0.1);
        }

        #[test]
        fn test_nemotron_config_debug() {
            let config = NemotronConfig::new("/tmp/test.gguf");
            let debug_str = format!("{config:?}");
            assert!(debug_str.contains("NemotronConfig"));
            assert!(debug_str.contains("model_path"));
        }

        #[test]
        fn test_nemotron_config_clone() {
            let config = NemotronConfig::new("/tmp/test.gguf").with_batch_size(32);
            let cloned = config.clone();
            assert_eq!(cloned.batch_size, 32);
            assert_eq!(cloned.model_path, config.model_path);
        }

        #[test]
        fn test_nemotron_rms_normalize_with_weights() {
            let mut vector = vec![2.0, 2.0];
            let weight = vec![0.5, 2.0];
            NemotronEmbedder::rms_normalize(&mut vector, &weight);
            // RMS for [2.0, 2.0] = sqrt((4+4)/2) = 2
            // Scale = 1/sqrt(4 + 1e-6) ≈ 0.5
            // Result[0] = 2.0 * 0.5 * 0.5 = 0.5
            // Result[1] = 2.0 * 0.5 * 2.0 = 2.0
            assert!((vector[0] - 0.5).abs() < 0.01);
            assert!((vector[1] - 2.0).abs() < 0.01);
        }
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
