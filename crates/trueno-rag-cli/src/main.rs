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
//!
//! # Index with recursive directory walking and subtitle support
//! trueno-rag index --path /data/ --output index/ --recursive
//!
//! # Index with timestamp-aware chunking for media transcripts
//! trueno-rag index --path /data/ --output index/ --recursive --chunk-strategy timestamp
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use trueno_rag::{
    chunk::{RecursiveChunker, TimestampChunker},
    embed::{Embedder, TfIdfEmbedder},
    fusion::FusionStrategy,
    loader::LoaderRegistry,
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

/// Chunking strategy selection
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum ChunkStrategy {
    /// Auto-select: TimestampChunker for media, RecursiveChunker for text
    #[default]
    Auto,
    /// Recursive character splitting (works for all content)
    Recursive,
    /// Timestamp-aware chunking (best for subtitle/transcript content)
    Timestamp,
}

/// Compute backend for transcription
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum BackendType {
    /// CPU with SIMD acceleration
    #[default]
    Cpu,
    /// GPU via wgpu (cross-platform)
    Gpu,
    /// NVIDIA CUDA (Linux/Windows)
    Cuda,
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

        /// Chunk size in characters (for recursive chunker)
        #[arg(long, default_value = "512")]
        chunk_size: usize,

        /// Chunk overlap in characters (for recursive chunker)
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

        /// Recursively scan subdirectories
        #[arg(short, long, default_value = "false")]
        recursive: bool,

        /// Chunking strategy (auto, recursive, timestamp)
        #[arg(long, value_enum, default_value = "auto")]
        chunk_strategy: ChunkStrategy,

        /// Write a JSON manifest of indexed files and chunks
        #[arg(long, default_value = "false")]
        manifest: bool,
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

    /// Batch transcribe media files to .srt sidecars
    Transcribe {
        /// Path to directory containing media files
        #[arg(short, long)]
        path: String,

        /// Recursively scan subdirectories
        #[arg(short, long, default_value = "false")]
        recursive: bool,

        /// Skip files that already have .srt/.vtt sidecars
        #[arg(long, default_value = "true")]
        skip_existing: bool,

        /// Number of parallel transcription jobs (CPU mode)
        #[arg(short, long, default_value = "1")]
        jobs: usize,

        /// Path to Whisper .apr model file (e.g. base.apr, large-v3-turbo.apr)
        #[arg(short, long)]
        model: Option<String>,

        /// Compute backend (cpu, gpu, cuda)
        #[arg(short, long, value_enum, default_value = "cpu")]
        backend: BackendType,

        /// Only report what would be transcribed (dry run)
        #[arg(long, default_value = "false")]
        dry_run: bool,
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
    /// Timestamp metadata for media-derived chunks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    start_secs: Option<f64>,
    /// Timestamp metadata for media-derived chunks
    #[serde(default, skip_serializing_if = "Option::is_none")]
    end_secs: Option<f64>,
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
            recursive,
            chunk_strategy,
            manifest,
        } => run_index(
            &path,
            &output,
            chunk_size,
            chunk_overlap,
            dimension,
            embedder,
            model,
            recursive,
            chunk_strategy,
            manifest,
        )?,
        Commands::Query {
            query,
            index,
            top_k,
            format,
        } => run_query(&query, &index, top_k, &format)?,
        Commands::Transcribe {
            path,
            recursive,
            skip_existing,
            jobs,
            model,
            backend,
            dry_run,
        } => run_transcribe(&path, recursive, skip_existing, jobs, model.as_deref(), backend, dry_run)?,
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
    println!("  - Chunkers: Recursive, Fixed, Sentence, Paragraph, Semantic, Structural, Timestamp");
    #[cfg(feature = "embeddings")]
    println!("  - Embedders: TF-IDF, FastEmbed (semantic) ✓");
    #[cfg(not(feature = "embeddings"))]
    println!("  - Embedders: TF-IDF (trainable), Mock (testing)");
    println!("  - Fusion: RRF, Linear, DBSF, Convex, Union, Intersection");
    println!("  - Rerankers: Lexical, CrossEncoder (mock), Composite");
    println!();
    println!("Supported formats:");
    let registry = LoaderRegistry::new();
    let exts: Vec<&str> = registry.supported_extensions();
    println!("  {}", exts.join(", "));
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

/// Discover media files in a directory.
fn discover_media_files(root: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if root.is_file() {
        if is_media_file(root) {
            files.push(root.to_path_buf());
        } else {
            anyhow::bail!("Not a media file: {}", root.display());
        }
        return Ok(files);
    }

    let mut dirs_to_visit = vec![root.to_path_buf()];
    while let Some(dir) = dirs_to_visit.pop() {
        let entries = fs::read_dir(&dir)
            .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

        for entry in entries {
            let path = entry?.path();
            if path.is_dir() && recursive {
                dirs_to_visit.push(path);
            } else if path.is_file() && is_media_file(&path) {
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

/// Check if a file has a media extension.
fn is_media_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| MEDIA_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
}

/// Classify media files into those with/without existing sidecars.
fn classify_media_sidecar_status(files: &[PathBuf]) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut has_sidecar = Vec::new();
    let mut needs_transcription = Vec::new();

    for file in files {
        if LoaderRegistry::find_sidecar(file).is_some() {
            has_sidecar.push(file.clone());
        } else {
            needs_transcription.push(file.clone());
        }
    }

    (has_sidecar, needs_transcription)
}

/// Transcription manifest for resume support.
#[derive(Serialize, Deserialize, Default)]
struct TranscribeManifest {
    /// Files that have been successfully transcribed.
    completed: Vec<String>,
    /// Files that failed transcription.
    failed: Vec<String>,
}

impl TranscribeManifest {
    fn load(root: &Path) -> Self {
        let manifest_path = root.join(".transcribe-manifest.json");
        fs::read_to_string(manifest_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    #[allow(dead_code)]
    fn save(&self, root: &Path) -> Result<()> {
        let manifest_path = root.join(".transcribe-manifest.json");
        let json = serde_json::to_string_pretty(self)?;
        fs::write(manifest_path, json)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_transcribe(
    path: &str,
    recursive: bool,
    skip_existing: bool,
    _jobs: usize,
    _model: Option<&str>,
    _backend: BackendType,
    dry_run: bool,
) -> Result<()> {
    let root = Path::new(path);

    if !root.exists() {
        anyhow::bail!("Path not found: {}", root.display());
    }

    // Stage 1: Discovery
    let start_time = std::time::Instant::now();
    println!("Discovering media files...");
    let media_files = discover_media_files(root, recursive)?;

    if media_files.is_empty() {
        println!("No media files found at: {}", root.display());
        return Ok(());
    }

    let ext_counts = classify_files(&media_files);
    println!(
        "Found {} media files{}",
        media_files.len(),
        if recursive { " (recursive)" } else { "" }
    );
    for (ext, count) in &ext_counts {
        println!("  {} .{} files", count, ext);
    }

    // Stage 2: Sidecar check + manifest resume
    let (has_sidecar, needs_transcription) = classify_media_sidecar_status(&media_files);
    let manifest = TranscribeManifest::load(root);
    let previously_completed = manifest.completed.len();

    // Filter out files already in the manifest (resume support)
    let to_process: Vec<PathBuf> = if skip_existing {
        needs_transcription
            .into_iter()
            .filter(|f| {
                !manifest
                    .completed
                    .contains(&f.to_string_lossy().to_string())
            })
            .collect()
    } else {
        media_files
            .iter()
            .filter(|f| {
                !manifest
                    .completed
                    .contains(&f.to_string_lossy().to_string())
            })
            .cloned()
            .collect()
    };

    println!(
        "\nSidecar status: {} with .srt/.vtt, {} need transcription",
        has_sidecar.len(),
        to_process.len()
    );
    if previously_completed > 0 {
        println!("  {} previously completed (from manifest)", previously_completed);
    }

    if to_process.is_empty() {
        println!("All files already have sidecars. Nothing to do.");
        return Ok(());
    }

    if dry_run {
        println!("\nDry run — files that would be transcribed:");
        for file in &to_process {
            println!("  {}", file.display());
        }
        println!("\nTotal: {} files", to_process.len());
        return Ok(());
    }

    // Stage 3: Transcription (feature-gated)
    #[cfg(feature = "transcription")]
    {
        run_transcription_batch(&to_process, _jobs, _model, _backend, root)?;
    }
    #[cfg(not(feature = "transcription"))]
    {
        println!(
            "\nTranscription requires the 'transcription' feature.\n\
             Build with: cargo build --release --features transcription\n\n\
             {} files need transcription. Run with --dry-run to list them.",
            to_process.len()
        );
    }

    // Throughput reporting
    let elapsed = start_time.elapsed();
    println!(
        "\nTotal time: {:.1}s ({:.1} files/sec)",
        elapsed.as_secs_f64(),
        media_files.len() as f64 / elapsed.as_secs_f64().max(0.001)
    );

    Ok(())
}

/// Run transcription on a batch of media files (requires transcription feature).
#[cfg(feature = "transcription")]
fn run_transcription_batch(
    files: &[PathBuf],
    _jobs: usize,
    model: Option<&str>,
    _backend: BackendType,
    root: &Path,
) -> Result<()> {
    use trueno_rag::{TranscriptionBackend, TranscriptionConfig, TranscriptionLoader};

    let backend = match _backend {
        BackendType::Cpu => TranscriptionBackend::Cpu,
        BackendType::Gpu => TranscriptionBackend::Gpu,
        BackendType::Cuda => TranscriptionBackend::Cuda,
    };

    let config = TranscriptionConfig {
        model_path: model.map(PathBuf::from),
        backend,
        ..TranscriptionConfig::default()
    };
    let loader = TranscriptionLoader::new(config);

    if loader.has_model() {
        println!("\nWhisper model loaded. Transcribing {} files...", files.len());
    } else {
        println!(
            "\nNo model specified (use --model <path.apr>). \
             Only files with sidecars will be loaded."
        );
    }

    let batch_start = std::time::Instant::now();
    let mut manifest = TranscribeManifest::load(root);
    let mut success = 0usize;
    let mut errors = 0usize;

    for (i, file) in files.iter().enumerate() {
        let filename = file.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        print!("  [{}/{}] {} ... ", i + 1, files.len(), filename);

        match loader.load(file) {
            Ok(_doc) => {
                success += 1;
                manifest.completed.push(file.to_string_lossy().to_string());
                println!("ok");
            }
            Err(e) => {
                errors += 1;
                manifest.failed.push(file.to_string_lossy().to_string());
                println!("FAILED: {e}");
            }
        }

        // Persist manifest every 10 files for resume support
        if (i + 1) % 10 == 0 {
            let _ = manifest.save(root);
        }
    }

    // Final manifest save
    manifest.save(root)?;

    let elapsed = batch_start.elapsed();
    println!(
        "\nComplete: {} succeeded, {} failed out of {} total ({:.1}s, {:.1} files/sec)",
        success,
        errors,
        files.len(),
        elapsed.as_secs_f64(),
        files.len() as f64 / elapsed.as_secs_f64().max(0.001),
    );

    Ok(())
}

/// Media file extensions that can be transcribed.
const MEDIA_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", // video
    "mp3", "wav", "flac", "ogg", "m4a", "aac", "wma", // audio
];

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

/// Discover files from a path using the loader registry.
fn discover_files(
    root: &Path,
    recursive: bool,
    registry: &LoaderRegistry,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if root.is_file() {
        if registry.loader_for(root).is_some() {
            files.push(root.to_path_buf());
        } else {
            anyhow::bail!(
                "Unsupported file format: {}",
                root.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("(none)")
            );
        }
        return Ok(files);
    }

    let mut dirs_to_visit = vec![root.to_path_buf()];
    while let Some(dir) = dirs_to_visit.pop() {
        let entries = fs::read_dir(&dir)
            .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

        for entry in entries {
            let path = entry?.path();
            if path.is_dir() && recursive {
                dirs_to_visit.push(path);
            } else if path.is_file() && registry.loader_for(&path).is_some() {
                files.push(path);
            }
        }
    }

    // Sort for deterministic ordering
    files.sort();
    Ok(files)
}

/// Classify discovered files by format for progress reporting.
fn classify_files(files: &[PathBuf]) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for file in files {
        let ext = file
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("other")
            .to_lowercase();
        *counts.entry(ext).or_insert(0) += 1;
    }
    counts
}

/// Load documents from discovered files, reporting progress and errors.
fn load_documents(
    files: &[PathBuf],
    registry: &LoaderRegistry,
) -> Result<Vec<Document>> {
    let mut documents = Vec::new();
    let mut load_errors = 0usize;

    for (i, file) in files.iter().enumerate() {
        match registry.load(file) {
            Ok(doc) => documents.push(doc),
            Err(e) => {
                eprintln!("  Warning: failed to load {}: {}", file.display(), e);
                load_errors += 1;
            }
        }
        if (i + 1) % 100 == 0 {
            println!("  Loaded {}/{} files...", i + 1, files.len());
        }
    }

    if documents.is_empty() {
        anyhow::bail!("All files failed to load ({} errors)", load_errors);
    }

    if load_errors > 0 {
        println!(
            "Loaded {} documents ({} failed)",
            documents.len(),
            load_errors
        );
    } else {
        println!("Loaded {} documents", documents.len());
    }

    Ok(documents)
}

/// Chunk documents and compute embeddings, returning parallel vectors.
fn chunk_and_embed(
    documents: &[Document],
    embedder: &dyn Embedder,
    recursive_chunker: &RecursiveChunker,
    timestamp_chunker: &TimestampChunker,
    strategy: ChunkStrategy,
) -> Result<(Vec<PersistedChunk>, Vec<Vec<f32>>)> {
    let mut all_chunks = Vec::new();
    let mut all_embeddings = Vec::new();

    for doc in documents {
        let use_timestamps = match strategy {
            ChunkStrategy::Timestamp => true,
            ChunkStrategy::Recursive => false,
            ChunkStrategy::Auto => doc.metadata.contains_key("subtitle_cues"),
        };
        let chunks: Vec<Chunk> = if use_timestamps {
            timestamp_chunker.chunk(doc)?
        } else {
            recursive_chunker.chunk(doc)?
        };

        for chunk in chunks {
            all_embeddings.push(embedder.embed(&chunk.content)?);
            all_chunks.push(PersistedChunk {
                content: chunk.content.clone(),
                title: chunk.metadata.title.clone(),
                source: doc.source.clone(),
                start_secs: chunk.metadata.custom.get("start_secs").and_then(serde_json::Value::as_f64),
                end_secs: chunk.metadata.custom.get("end_secs").and_then(serde_json::Value::as_f64),
            });
        }
    }

    Ok((all_chunks, all_embeddings))
}

#[allow(clippy::too_many_arguments)]
fn run_index(
    path: &str,
    output: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    dimension: usize,
    embedder_type: EmbedderType,
    #[allow(unused_variables)] model: SemanticModel,
    recursive: bool,
    chunk_strategy: ChunkStrategy,
    manifest: bool,
) -> Result<()> {
    let path = Path::new(path);

    // Validate path exists
    if !path.exists() {
        anyhow::bail!("Path not found: {}", path.display());
    }

    // Discover files via LoaderRegistry
    let registry = LoaderRegistry::new();
    let files = discover_files(path, recursive, &registry)?;

    if files.is_empty() {
        let exts = registry.supported_extensions().join(", ");
        anyhow::bail!(
            "No supported files found at: {} (supported: {})",
            path.display(),
            exts
        );
    }

    // Report discovery
    let classification = classify_files(&files);
    println!(
        "Scanning {}{}... found {} files",
        path.display(),
        if recursive { " (recursive)" } else { "" },
        files.len()
    );
    for (ext, count) in &classification {
        println!("  {} .{} files", count, ext);
    }

    let documents = load_documents(&files, &registry)?;

    // Determine which documents have timestamp metadata
    let media_count = documents
        .iter()
        .filter(|d| d.metadata.contains_key("subtitle_cues"))
        .count();
    let text_count = documents.len() - media_count;
    if media_count > 0 {
        println!("  {} with timestamps, {} plain text", media_count, text_count);
    }

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

    // Chunk and embed all documents
    let recursive_chunker = RecursiveChunker::new(chunk_size, chunk_overlap);
    let timestamp_chunker = TimestampChunker::new(60.0);
    let (all_chunks, all_embeddings) = chunk_and_embed(
        &documents,
        &*embedder_box,
        &recursive_chunker,
        &timestamp_chunker,
        chunk_strategy,
    )?;

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

    // Write manifest if requested
    if manifest {
        let manifest_data = build_index_manifest(&files, &classification, &persisted.chunks);
        let manifest_file = output_path.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&manifest_data)?;
        fs::write(&manifest_file, manifest_json)?;
        println!("Manifest saved to: {}", manifest_file.display());
    }

    Ok(())
}

/// Build a JSON manifest of indexed files and chunks.
fn build_index_manifest(
    files: &[PathBuf],
    classification: &HashMap<String, usize>,
    chunks: &[PersistedChunk],
) -> serde_json::Value {
    let file_entries: Vec<serde_json::Value> = files
        .iter()
        .map(|f| {
            let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("unknown");
            serde_json::json!({
                "path": f.to_string_lossy(),
                "extension": ext,
            })
        })
        .collect();

    serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "file_count": files.len(),
        "chunk_count": chunks.len(),
        "format_counts": classification,
        "files": file_entries,
    })
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

    format_query_results(query, &scores, &persisted.chunks, format)
}

/// Format and print query results in text or JSON format.
fn format_query_results(
    query: &str,
    scores: &[(usize, f32)],
    chunks: &[PersistedChunk],
    format: &str,
) -> Result<()> {
    if format == "json" {
        let results: Vec<serde_json::Value> = scores
            .iter()
            .enumerate()
            .map(|(rank, (i, score))| {
                let chunk = &chunks[*i];
                let mut result = serde_json::json!({
                    "rank": rank + 1,
                    "score": score,
                    "content": chunk.content,
                    "title": chunk.title,
                    "source": chunk.source,
                });
                if let Some(start) = chunk.start_secs {
                    result["start_secs"] = serde_json::json!(start);
                }
                if let Some(end) = chunk.end_secs {
                    result["end_secs"] = serde_json::json!(end);
                }
                result
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        println!("Query: \"{query}\"\n");
        println!("Results ({}):", scores.len());
        println!("{}", "-".repeat(50));

        for (rank, (i, score)) in scores.iter().enumerate() {
            let chunk = &chunks[*i];
            let title = chunk.title.as_deref().unwrap_or("Untitled");
            let time_info = match (chunk.start_secs, chunk.end_secs) {
                (Some(start), Some(end)) => format!(
                    " [{}–{}]",
                    trueno_rag::media::format_display_time(start),
                    trueno_rag::media::format_display_time(end),
                ),
                _ => String::new(),
            };
            println!("{}. [Score: {:.3}] {}{}", rank + 1, score, title, time_info);
            let preview = &chunk.content[..80.min(chunk.content.len())];
            println!("   {preview}...\n");
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

    #[test]
    fn test_discover_files_single_file() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_discover_single");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("test.txt");
        fs::write(&file, "hello").unwrap();

        let registry = LoaderRegistry::new();
        let files = discover_files(&file, false, &registry).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], file);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_files_directory_non_recursive() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_discover_dir");
        let sub = dir.join("sub");
        let _ = fs::create_dir_all(&sub);
        fs::write(dir.join("a.txt"), "a").unwrap();
        fs::write(dir.join("b.md"), "b").unwrap();
        fs::write(dir.join("c.mp4"), "c").unwrap(); // unsupported
        fs::write(sub.join("d.txt"), "d").unwrap(); // in subdir

        let registry = LoaderRegistry::new();
        let files = discover_files(&dir, false, &registry).unwrap();
        // Only a.txt and b.md in top-level (not c.mp4, not sub/d.txt)
        assert_eq!(files.len(), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_files_recursive() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_discover_recursive");
        let sub = dir.join("sub");
        let deep = sub.join("deep");
        let _ = fs::create_dir_all(&deep);
        fs::write(dir.join("a.txt"), "a").unwrap();
        fs::write(sub.join("b.srt"), "1\n00:00:01,000 --> 00:00:02,000\nb\n").unwrap();
        fs::write(deep.join("c.md"), "c").unwrap();

        let registry = LoaderRegistry::new();
        let files = discover_files(&dir, true, &registry).unwrap();
        assert_eq!(files.len(), 3);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_files_unsupported_single() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_discover_unsup");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("video.mp4");
        fs::write(&file, "fake").unwrap();

        let registry = LoaderRegistry::new();
        let result = discover_files(&file, false, &registry);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_classify_files() {
        let files = vec![
            PathBuf::from("/data/a.txt"),
            PathBuf::from("/data/b.txt"),
            PathBuf::from("/data/c.srt"),
            PathBuf::from("/data/d.md"),
        ];
        let counts = classify_files(&files);
        assert_eq!(counts["txt"], 2);
        assert_eq!(counts["srt"], 1);
        assert_eq!(counts["md"], 1);
    }

    #[test]
    fn test_chunk_strategy_default() {
        // Ensure Auto is the default
        let strategy = ChunkStrategy::default();
        assert!(matches!(strategy, ChunkStrategy::Auto));
    }
}
