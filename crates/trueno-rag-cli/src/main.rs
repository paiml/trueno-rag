//! Trueno-RAG CLI
//!
//! Command-line interface for the Trueno-RAG pipeline.
//!
//! ## Features
//!
//! - `embeddings` - Enable production semantic embeddings via fastembed (ONNX Runtime)
//!
//! ## Usage
//!
//! ```bash
//! # Build with semantic embeddings support
//! cargo build --release --features embeddings
//!
//! # Index documents with semantic embeddings
//! trueno-rag index --path docs/ --output index/ --embedder semantic
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use trueno_rag::{
    chunk::RecursiveChunker,
    embed::{Embedder, TfIdfEmbedder},
    fusion::FusionStrategy,
    pipeline::RagPipelineBuilder,
    rerank::LexicalReranker,
    Chunk, Chunker, Document,
};

#[cfg(feature = "embeddings")]
use trueno_rag::{EmbeddingModelType, FastEmbedder};

/// Embedder type selection
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum EmbedderType {
    /// TF-IDF statistical embeddings (default, no downloads)
    #[default]
    Tfidf,
    /// Semantic embeddings via fastembed (requires `embeddings` feature)
    Semantic,
}

/// Model selection for semantic embeddings
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum SemanticModel {
    /// all-MiniLM-L6-v2: Fast, good quality (384 dims)
    #[default]
    MiniLm,
    /// BGE-small-en-v1.5: Balanced performance (384 dims)
    BgeSmall,
    /// BGE-base-en-v1.5: Higher quality (768 dims)
    BgeBase,
}

#[derive(Parser)]
#[command(name = "trueno-rag")]
#[command(author = "Pragmatic AI Labs")]
#[command(version)]
#[command(about = "Pure-Rust RAG pipeline CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a demo RAG query
    Demo {
        /// Query string
        #[arg(short, long, default_value = "What is machine learning?")]
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value = "3")]
        top_k: usize,
    },

    /// Index documents from a file or directory
    Index {
        /// Path to document(s)
        #[arg(short, long)]
        path: String,

        /// Output directory for index
        #[arg(short, long)]
        output: String,

        /// Chunk size in characters
        #[arg(long, default_value = "512")]
        chunk_size: usize,

        /// Chunk overlap in characters
        #[arg(long, default_value = "64")]
        chunk_overlap: usize,

        /// Embedding dimension (only for tfidf embedder)
        #[arg(long, default_value = "256")]
        dimension: usize,

        /// Embedder type (tfidf or semantic)
        #[arg(short, long, value_enum, default_value = "tfidf")]
        embedder: EmbedderType,

        /// Model for semantic embeddings (mini-lm, bge-small, bge-base)
        #[arg(short, long, value_enum, default_value = "mini-lm")]
        model: SemanticModel,
    },

    /// Query the RAG pipeline
    Query {
        /// Query string
        query: String,

        /// Path to index directory
        #[arg(short, long)]
        index: String,

        /// Number of results
        #[arg(short, long, default_value = "5")]
        top_k: usize,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Show pipeline info
    Info,
}

/// Persisted index structure
#[derive(Serialize, Deserialize)]
struct PersistedIndex {
    chunks: Vec<PersistedChunk>,
    embeddings: Vec<Vec<f32>>,
    dimension: usize,
    /// Embedder type used (for query compatibility)
    #[serde(default)]
    embedder_type: String,
    /// Model name (for semantic embeddings)
    #[serde(default)]
    model_name: Option<String>,
}

/// Persisted chunk data
#[derive(Serialize, Deserialize)]
struct PersistedChunk {
    content: String,
    title: Option<String>,
    source: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Demo { query, top_k } => run_demo(&query, top_k)?,
        Commands::Index {
            path,
            output,
            chunk_size,
            chunk_overlap,
            dimension,
            embedder,
            model,
        } => run_index(
            &path,
            &output,
            chunk_size,
            chunk_overlap,
            dimension,
            embedder,
            model,
        )?,
        Commands::Query {
            query,
            index,
            top_k,
            format,
        } => run_query(&query, &index, top_k, &format)?,
        Commands::Info => run_info(),
    }

    Ok(())
}

fn run_info() {
    println!("Trueno-RAG Pipeline");
    println!("==================");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Components:");
    println!("  - Chunkers: Recursive, Fixed, Sentence, Paragraph, Semantic, Structural");
    #[cfg(feature = "embeddings")]
    println!("  - Embedders: TF-IDF, FastEmbed (semantic) ✓");
    #[cfg(not(feature = "embeddings"))]
    println!("  - Embedders: TF-IDF (trainable), Mock (testing)");
    println!("  - Fusion: RRF, Linear, DBSF, Convex, Union, Intersection");
    println!("  - Rerankers: Lexical, CrossEncoder (mock), Composite");
    println!();
    #[cfg(feature = "embeddings")]
    {
        println!("Semantic Embedding Models:");
        println!("  - mini-lm: sentence-transformers/all-MiniLM-L6-v2 (384 dims, fast)");
        println!("  - bge-small: BAAI/bge-small-en-v1.5 (384 dims, balanced)");
        println!("  - bge-base: BAAI/bge-base-en-v1.5 (768 dims, quality)");
    }
    #[cfg(not(feature = "embeddings"))]
    {
        println!("Note: Build with --features embeddings for semantic search");
    }
}

fn run_demo(query: &str, top_k: usize) -> Result<()> {
    println!("=== Trueno-RAG Demo ===\n");

    // Sample documents for training TF-IDF
    let sample_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "Deep learning uses neural networks with many layers to learn representations of data. It has achieved breakthrough results in image and speech recognition.",
        "Natural language processing enables computers to understand, interpret, and generate human language in a valuable way.",
        "Retrieval-Augmented Generation combines retrieval systems with generative models to produce more accurate and grounded responses.",
    ];

    // Train TF-IDF embedder
    let mut embedder = TfIdfEmbedder::new(128);
    let refs: Vec<&str> = sample_texts.iter().map(AsRef::as_ref).collect();
    embedder.fit(&refs);

    // Build pipeline
    let mut pipeline = RagPipelineBuilder::new()
        .chunker(RecursiveChunker::new(256, 32))
        .embedder(embedder)
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .max_context_tokens(2000)
        .build()?;

    // Create documents
    let docs = vec![
        Document::new(sample_texts[0]).with_title("Machine Learning Basics"),
        Document::new(sample_texts[1]).with_title("Deep Learning Overview"),
        Document::new(sample_texts[2]).with_title("NLP Introduction"),
        Document::new(sample_texts[3]).with_title("RAG Systems"),
    ];

    // Index
    let chunk_count = pipeline.index_documents(&docs)?;
    println!(
        "Indexed {} documents ({} chunks)\n",
        docs.len(),
        chunk_count
    );

    // Query
    println!("Query: \"{}\"\n", query);

    let (results, context) = pipeline.query_with_context(query, top_k)?;

    println!("Results ({}):", results.len());
    println!("{}", "-".repeat(50));

    for (i, result) in results.iter().enumerate() {
        let title = result.chunk.metadata.title.as_deref().unwrap_or("Untitled");
        println!("{}. [Score: {:.3}] {}", i + 1, result.best_score(), title);
        let preview = &result.chunk.content[..80.min(result.chunk.content.len())];
        println!("   {}...\n", preview);
    }

    println!("{}", "=".repeat(50));
    println!("Assembled Context:\n");
    println!("{}", context.format_with_citations());

    println!("\nCitations:");
    println!("{}", context.citation_list());

    Ok(())
}

fn run_index(
    path: &str,
    output: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    dimension: usize,
    embedder_type: EmbedderType,
    #[allow(unused_variables)] model: SemanticModel,
) -> Result<()> {
    let path = Path::new(path);

    // Validate path exists
    if !path.exists() {
        anyhow::bail!("Path not found: {}", path.display());
    }

    // Collect documents
    let mut documents = Vec::new();

    if path.is_file() {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;
        let title = path.file_name().and_then(|n| n.to_str()).map(String::from);
        documents.push(
            Document::new(&content)
                .with_title(title.as_deref().unwrap_or("Untitled"))
                .with_source(path.to_string_lossy()),
        );
    } else if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();
            if file_path.is_file() {
                if let Some(ext) = file_path.extension() {
                    if ext == "txt" || ext == "md" {
                        let content = fs::read_to_string(&file_path)?;
                        let title = file_path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .map(String::from);
                        documents.push(
                            Document::new(&content)
                                .with_title(title.as_deref().unwrap_or("Untitled"))
                                .with_source(file_path.to_string_lossy()),
                        );
                    }
                }
            }
        }
    }

    if documents.is_empty() {
        anyhow::bail!("No documents found at path: {}", path.display());
    }

    println!("Found {} documents", documents.len());

    // Create embedder based on selection
    let (embedder_box, actual_dimension, embedder_name, model_name): (
        Box<dyn Embedder>,
        usize,
        String,
        Option<String>,
    ) = match embedder_type {
        EmbedderType::Tfidf => {
            let mut embedder = TfIdfEmbedder::new(dimension);
            let doc_texts: Vec<&str> = documents.iter().map(|d| d.content.as_str()).collect();
            embedder.fit(&doc_texts);
            println!("Using TF-IDF embedder (dimension: {})", dimension);
            (Box::new(embedder), dimension, "tfidf".to_string(), None)
        }
        EmbedderType::Semantic => {
            #[cfg(feature = "embeddings")]
            {
                let model_type = match model {
                    SemanticModel::MiniLm => EmbeddingModelType::AllMiniLmL6V2,
                    SemanticModel::BgeSmall => EmbeddingModelType::BgeSmallEnV15,
                    SemanticModel::BgeBase => EmbeddingModelType::BgeBaseEnV15,
                };
                println!(
                    "Loading semantic model: {} (dimension: {})",
                    model_type.model_name(),
                    model_type.dimension()
                );
                let embedder = FastEmbedder::new(model_type)
                    .context("Failed to initialize semantic embedder")?;
                let dim = embedder.dimension();
                let name = model_type.model_name().to_string();
                (Box::new(embedder), dim, "semantic".to_string(), Some(name))
            }
            #[cfg(not(feature = "embeddings"))]
            {
                anyhow::bail!(
                    "Semantic embeddings require the 'embeddings' feature.\n\
                     Build with: cargo build --features embeddings"
                );
            }
        }
    };

    // Chunk documents
    let chunker = RecursiveChunker::new(chunk_size, chunk_overlap);
    let mut all_chunks: Vec<PersistedChunk> = Vec::new();
    let mut all_embeddings: Vec<Vec<f32>> = Vec::new();

    for doc in &documents {
        let chunks: Vec<Chunk> = chunker.chunk(doc)?;
        for chunk in chunks {
            let embedding = embedder_box.embed(&chunk.content)?;
            all_chunks.push(PersistedChunk {
                content: chunk.content.clone(),
                title: chunk.metadata.title.clone(),
                source: doc.source.clone(),
            });
            all_embeddings.push(embedding);
        }
    }

    println!(
        "Indexed {} documents ({} chunks)",
        documents.len(),
        all_chunks.len()
    );

    // Create persisted index
    let persisted = PersistedIndex {
        chunks: all_chunks,
        embeddings: all_embeddings,
        dimension: actual_dimension,
        embedder_type: embedder_name,
        model_name,
    };

    // Save index
    let output_path = Path::new(output);
    fs::create_dir_all(output_path)?;

    let index_file = output_path.join("index.json");
    let json = serde_json::to_string_pretty(&persisted)?;
    fs::write(&index_file, json)?;

    println!("Index saved to: {}", index_file.display());

    Ok(())
}

fn run_query(query: &str, index_path: &str, top_k: usize, format: &str) -> Result<()> {
    let index_path = Path::new(index_path);
    let index_file = index_path.join("index.json");

    if !index_file.exists() {
        anyhow::bail!("Index not found at: {}", index_file.display());
    }

    // Load index
    let json = fs::read_to_string(&index_file)?;
    let persisted: PersistedIndex = serde_json::from_str(&json)?;

    // Create embedder based on index type
    let query_embedding: Vec<f32> = if persisted.embedder_type == "semantic" {
        #[cfg(feature = "embeddings")]
        {
            // Determine model from stored name or default
            let model_type = match persisted.model_name.as_deref() {
                Some(name) if name.contains("bge-base") => EmbeddingModelType::BgeBaseEnV15,
                Some(name) if name.contains("bge-small") => EmbeddingModelType::BgeSmallEnV15,
                _ => EmbeddingModelType::AllMiniLmL6V2, // Default
            };
            println!(
                "Using semantic embedder: {} (dimension: {})",
                model_type.model_name(),
                model_type.dimension()
            );
            let embedder = FastEmbedder::new(model_type)
                .context("Failed to initialize semantic embedder for query")?;
            embedder.embed(query)?
        }
        #[cfg(not(feature = "embeddings"))]
        {
            anyhow::bail!(
                "This index uses semantic embeddings.\n\
                 Build with: cargo build --features embeddings"
            );
        }
    } else {
        // TF-IDF: rebuild from chunk content
        let mut embedder = TfIdfEmbedder::new(persisted.dimension);
        let refs: Vec<&str> = persisted
            .chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect();
        embedder.fit(&refs);
        embedder.embed(query)?
    };

    // Compute similarities
    let mut scores: Vec<(usize, f32)> = persisted
        .embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| {
            let sim = cosine_similarity(&query_embedding, emb);
            (i, sim)
        })
        .collect();

    // Sort by score descending
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);

    // Format output
    if format == "json" {
        let results: Vec<serde_json::Value> = scores
            .iter()
            .enumerate()
            .map(|(rank, (i, score))| {
                serde_json::json!({
                    "rank": rank + 1,
                    "score": score,
                    "content": persisted.chunks[*i].content,
                    "title": persisted.chunks[*i].title,
                    "source": persisted.chunks[*i].source,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        println!("Query: \"{}\"\n", query);
        println!("Results ({}):", scores.len());
        println!("{}", "-".repeat(50));

        for (rank, (i, score)) in scores.iter().enumerate() {
            let chunk = &persisted.chunks[*i];
            let title = chunk.title.as_deref().unwrap_or("Untitled");
            println!("{}. [Score: {:.3}] {}", rank + 1, score, title);
            let preview = &chunk.content[..80.min(chunk.content.len())];
            println!("   {}...\n", preview);
        }
    }

    Ok(())
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
