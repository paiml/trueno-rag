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
use globset::{Glob, GlobSet, GlobSetBuilder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
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

        /// Number of parallel loading jobs
        #[arg(short, long, default_value = "1")]
        jobs: usize,

        /// Write a JSON manifest of indexed files and chunks
        #[arg(long, default_value = "false")]
        manifest: bool,

        /// Glob patterns to exclude files/directories (repeatable)
        #[arg(long)]
        exclude: Vec<String>,

        /// Deduplicate chunks with identical content (keeps first occurrence)
        #[arg(long, default_value = "false")]
        dedup: bool,

        /// Also export a SQLite+FTS5 index (requires sqlite feature)
        #[arg(long, default_value = "false")]
        sqlite: bool,
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

        /// Retrieval mode: dense, sparse (BM25), hybrid (BM25 + dense RRF)
        #[arg(long, default_value = "hybrid")]
        mode: String,

        /// Fusion strategy (hybrid mode only): rrf, linear, dbsf
        #[arg(long, default_value = "rrf")]
        fusion: String,

        /// Fusion parameter: RRF k value or Linear dense_weight
        #[arg(long)]
        fusion_k: Option<f32>,

        /// Candidates per source for hybrid retrieval
        #[arg(long, default_value = "50")]
        candidates: usize,

        /// Reranking strategy: none, lexical
        #[arg(long, default_value = "none")]
        rerank: String,
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

        /// Initial prompt to condition decoder vocabulary (e.g. "AWS, Kubernetes, YAML")
        #[arg(long)]
        prompt: Option<String>,

        /// Path to file with hotwords (one per line) to boost during decoding
        #[arg(long)]
        hotwords: Option<String>,

        /// Glob patterns to exclude files/directories (repeatable)
        #[arg(long)]
        exclude: Vec<String>,
    },

    /// Show pipeline info
    Info,

    /// Evaluation framework: generate ground truth, run retrieval, judge relevance
    #[cfg(feature = "eval")]
    Eval {
        #[command(subcommand)]
        action: EvalAction,
    },
}

/// Eval sub-subcommands
#[cfg(feature = "eval")]
#[derive(Subcommand)]
enum EvalAction {
    /// Sample chunks from index for ground truth generation (no API needed)
    Sample {
        /// Path to index directory (containing index.json)
        #[arg(short, long)]
        index: String,

        /// Output path for sampled-chunks JSONL
        #[arg(short, long)]
        output: String,

        /// Number of chunks to sample
        #[arg(long, default_value = "250")]
        sample_size: usize,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Generate synthetic ground truth from an index via Claude API (requires ANTHROPIC_API_KEY)
    Generate {
        /// Path to index directory (containing index.json)
        #[arg(short, long)]
        index: String,

        /// Output path for ground-truth JSONL
        #[arg(short, long)]
        output: String,

        /// Number of query-chunk pairs to generate
        #[arg(long, default_value = "250")]
        sample_size: usize,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Claude model for question generation
        #[arg(long, default_value = "claude-sonnet-4-20250514")]
        model: String,

        /// Sample chunks only — no API calls (dry run)
        #[arg(long, default_value = "false")]
        dry_run: bool,
    },

    /// Run retrieval queries from ground truth and dump raw results
    Retrieve {
        /// Path to index directory
        #[arg(short, long)]
        index: String,

        /// Path to ground-truth JSONL
        #[arg(short, long)]
        ground_truth: String,

        /// Output path for retrieval results JSONL
        #[arg(short, long)]
        output: String,

        /// Number of results per query
        #[arg(long, default_value = "10")]
        top_k: usize,

        /// Retrieval mode: dense (TF-IDF only), sparse (BM25 only), hybrid (fused)
        #[arg(long, default_value = "dense")]
        mode: String,

        /// Fusion strategy (hybrid mode only): rrf, linear, dbsf
        #[arg(long, default_value = "rrf")]
        fusion: String,

        /// Fusion parameter: RRF k value or Linear dense_weight
        #[arg(long)]
        fusion_k: Option<f32>,

        /// Candidates per source for hybrid retrieval
        #[arg(long, default_value = "50")]
        candidates: usize,

        /// Reranking strategy: none, lexical
        #[arg(long, default_value = "none")]
        rerank: String,
    },

    /// Judge retrieval results for relevance via Claude API and compute metrics
    Judge {
        /// Path to retrieval-results JSONL
        #[arg(short, long)]
        retrieval_results: String,

        /// Path to ground-truth JSONL (for metadata)
        #[arg(short, long)]
        ground_truth: String,

        /// Output path for eval results JSON
        #[arg(short, long)]
        output: String,

        /// Path to judge cache JSON (created if absent)
        #[arg(long, default_value = "judge-cache.json")]
        cache: String,

        /// Number of results to judge per query
        #[arg(long, default_value = "10")]
        top_k: usize,

        /// Claude model for judging
        #[arg(long, default_value = "claude-sonnet-4-20250514")]
        model: String,
    },

    /// Compute IR metrics from pre-judged results (no API needed)
    Metrics {
        /// Path to retrieval-results JSONL
        #[arg(short, long)]
        retrieval_results: String,

        /// Path to judgments JSONL (produced by Claude Code or external judge)
        #[arg(short, long)]
        judgments: String,

        /// Output path for eval results JSON
        #[arg(short, long)]
        output: String,
    },

    /// Compare two eval result files
    Compare {
        /// Baseline results JSON
        #[arg(long)]
        baseline: String,

        /// Candidate results JSON
        #[arg(long)]
        candidate: String,
    },

    /// Regression gate — exit non-zero if below thresholds
    Gate {
        /// Path to eval results JSON
        #[arg(long)]
        results: String,

        /// Minimum MRR threshold
        #[arg(long, default_value = "0.50")]
        min_mrr: f64,

        /// Minimum Hit@5 threshold
        #[arg(long, default_value = "0.70")]
        min_hit5: f64,
    },
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
            jobs,
            manifest,
            exclude,
            dedup,
            sqlite,
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
            jobs,
            manifest,
            &exclude,
            dedup,
            sqlite,
        )?,
        Commands::Query {
            query,
            index,
            top_k,
            format,
            mode,
            fusion,
            fusion_k,
            candidates,
            rerank,
        } => run_query(&query, &index, top_k, &format, &mode, &fusion, fusion_k, candidates, &rerank)?,
        Commands::Transcribe {
            path,
            recursive,
            skip_existing,
            jobs,
            model,
            backend,
            dry_run,
            prompt,
            hotwords,
            exclude,
        } => run_transcribe(
            &path,
            recursive,
            skip_existing,
            jobs,
            model.as_deref(),
            backend,
            dry_run,
            prompt.as_deref(),
            hotwords.as_deref(),
            &exclude,
        )?,
        Commands::Info => run_info(),
        #[cfg(feature = "eval")]
        Commands::Eval { action } => run_eval(action)?,
    }

    Ok(())
}

fn run_info() {
    println!("Trueno-RAG Pipeline");
    println!("==================");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Components:");
    println!(
        "  - Chunkers: Recursive, Fixed, Sentence, Paragraph, Semantic, Structural, Timestamp"
    );
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
fn discover_media_files(
    root: &Path,
    recursive: bool,
    exclude: &Option<GlobSet>,
) -> Result<Vec<PathBuf>> {
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
    let mut excluded_count = 0usize;
    while let Some(dir) = dirs_to_visit.pop() {
        let entries = fs::read_dir(&dir)
            .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

        for entry in entries {
            let path = entry?.path();
            if is_excluded(&path, exclude) {
                excluded_count += 1;
                continue;
            }
            if path.is_dir() && recursive {
                dirs_to_visit.push(path);
            } else if path.is_file() && is_media_file(&path) {
                files.push(path);
            }
        }
    }

    if excluded_count > 0 {
        println!("Excluded {} paths by glob pattern", excluded_count);
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

/// Discover media files and print a summary of what was found.
fn discover_and_report_media(
    root: &Path,
    recursive: bool,
    exclude: &Option<GlobSet>,
) -> Result<Vec<PathBuf>> {
    println!("Discovering media files...");
    let media_files = discover_media_files(root, recursive, exclude)?;

    if media_files.is_empty() {
        println!("No media files found at: {}", root.display());
        return Ok(media_files);
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
    Ok(media_files)
}

/// Filter media files by sidecar status and manifest, returning files to process.
fn filter_files_for_transcription(
    media_files: &[PathBuf],
    root: &Path,
    skip_existing: bool,
) -> Vec<PathBuf> {
    let (has_sidecar, needs_transcription) = classify_media_sidecar_status(media_files);
    let manifest = TranscribeManifest::load(root);
    let previously_completed = manifest.completed.len();

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
        println!(
            "  {} previously completed (from manifest)",
            previously_completed
        );
    }
    to_process
}

/// Run the transcription feature gate (or print a message if not available).
#[allow(clippy::unnecessary_wraps)]
fn run_transcription_or_report(
    to_process: &[PathBuf],
    jobs: usize,
    model: Option<&str>,
    backend: BackendType,
    root: &Path,
    prompt: Option<&str>,
    hotwords: &[String],
) -> Result<()> {
    #[cfg(feature = "transcription")]
    {
        run_transcription_batch(to_process, jobs, model, backend, root, prompt, hotwords)?;
    }
    #[cfg(not(feature = "transcription"))]
    {
        let _ = (to_process, jobs, model, backend, root, prompt, hotwords);
        println!(
            "\nTranscription requires the 'transcription' feature.\n\
             Build with: cargo build --release --features transcription\n\n\
             {} files need transcription. Run with --dry-run to list them.",
            to_process.len()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_transcribe(
    path: &str,
    recursive: bool,
    skip_existing: bool,
    jobs: usize,
    model: Option<&str>,
    backend: BackendType,
    dry_run: bool,
    prompt: Option<&str>,
    hotwords_file: Option<&str>,
    exclude_patterns: &[String],
) -> Result<()> {
    let root = Path::new(path);
    if !root.exists() {
        anyhow::bail!("Path not found: {}", root.display());
    }

    let exclude = build_exclude_set(exclude_patterns)?;
    let start_time = std::time::Instant::now();
    let media_files = discover_and_report_media(root, recursive, &exclude)?;
    if media_files.is_empty() {
        return Ok(());
    }

    let to_process = filter_files_for_transcription(&media_files, root, skip_existing);
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

    // Load hotwords from file if provided (one word per line)
    let hotwords: Vec<String> = hotwords_file
        .map(|path| {
            fs::read_to_string(path)
                .unwrap_or_default()
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect()
        })
        .unwrap_or_default();

    if !hotwords.is_empty() {
        println!("Loaded {} hotwords for vocabulary biasing", hotwords.len());
    }
    if let Some(p) = prompt {
        println!("Using prompt: {:?}", p);
    }

    run_transcription_or_report(&to_process, jobs, model, backend, root, prompt, &hotwords)?;

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
    jobs: usize,
    model: Option<&str>,
    backend_type: BackendType,
    root: &Path,
    prompt: Option<&str>,
    hotwords: &[String],
) -> Result<()> {
    use trueno_rag::{
        DocumentLoader, TranscriptionBackend, TranscriptionConfig, TranscriptionLoader,
    };

    let backend = match backend_type {
        BackendType::Cpu => TranscriptionBackend::Cpu,
        BackendType::Gpu => TranscriptionBackend::Gpu,
        BackendType::Cuda => TranscriptionBackend::Cuda,
    };

    let config = TranscriptionConfig {
        model_path: model.map(PathBuf::from),
        backend,
        prompt: prompt.map(String::from),
        hotwords: hotwords.to_vec(),
        ..TranscriptionConfig::default()
    };
    let loader = TranscriptionLoader::new(config);

    if loader.has_model() {
        println!(
            "\nWhisper model loaded. Transcribing {} files...",
            files.len()
        );
    } else {
        println!(
            "\nNo model specified (use --model <path.apr>). \
             Only files with sidecars will be loaded."
        );
    }

    let batch_start = std::time::Instant::now();
    let manifest = Mutex::new(TranscribeManifest::load(root));
    let success = Mutex::new(0usize);
    let errors = Mutex::new(0usize);

    let process_file = |file: &PathBuf| {
        let filename = file.file_name().and_then(|n| n.to_str()).unwrap_or("?");

        match loader.load(file) {
            Ok(_doc) => {
                *success.lock().unwrap() += 1;
                manifest
                    .lock()
                    .unwrap()
                    .completed
                    .push(file.to_string_lossy().to_string());
                println!("  {} ... ok", filename);
            }
            Err(e) => {
                *errors.lock().unwrap() += 1;
                manifest
                    .lock()
                    .unwrap()
                    .failed
                    .push(file.to_string_lossy().to_string());
                println!("  {} ... FAILED: {e}", filename);
            }
        }
    };

    if jobs > 1 {
        println!("Using {} parallel transcription jobs", jobs);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .context("Failed to create thread pool")?;
        pool.install(|| {
            files.par_iter().for_each(process_file);
        });
    } else {
        files.iter().for_each(process_file);
    }

    // Final manifest save
    let manifest = manifest.into_inner().unwrap();
    manifest.save(root)?;

    let success = success.into_inner().unwrap();
    let errors = errors.into_inner().unwrap();
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

/// Build a GlobSet from exclude patterns. Returns None if no patterns.
fn build_exclude_set(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(
            Glob::new(pattern)
                .with_context(|| format!("Invalid exclude glob: {pattern}"))?,
        );
    }
    Ok(Some(builder.build().context("Failed to build exclude set")?))
}

/// Check if a path should be excluded by glob patterns.
fn is_excluded(path: &Path, exclude: &Option<GlobSet>) -> bool {
    match exclude {
        Some(set) => set.is_match(path),
        None => false,
    }
}

/// Discover files from a path using the loader registry.
fn discover_files(
    root: &Path,
    recursive: bool,
    registry: &LoaderRegistry,
    exclude: &Option<GlobSet>,
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
    let mut excluded_count = 0usize;
    while let Some(dir) = dirs_to_visit.pop() {
        let entries = fs::read_dir(&dir)
            .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

        for entry in entries {
            let path = entry?.path();
            if is_excluded(&path, exclude) {
                excluded_count += 1;
                continue;
            }
            if path.is_dir() && recursive {
                dirs_to_visit.push(path);
            } else if path.is_file() && registry.loader_for(&path).is_some() {
                files.push(path);
            }
        }
    }

    if excluded_count > 0 {
        println!("Excluded {} paths by glob pattern", excluded_count);
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
/// Uses rayon parallel loading when `jobs` > 1.
fn load_documents(
    files: &[PathBuf],
    registry: &LoaderRegistry,
    jobs: usize,
) -> Result<Vec<Document>> {
    if jobs > 1 {
        load_documents_parallel(files, registry, jobs)
    } else {
        load_documents_sequential(files, registry)
    }
}

fn load_documents_sequential(
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

    finish_load_report(documents, load_errors)
}

fn load_documents_parallel(
    files: &[PathBuf],
    registry: &LoaderRegistry,
    jobs: usize,
) -> Result<Vec<Document>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .context("Failed to create thread pool")?;

    let documents = Mutex::new(Vec::new());
    let load_errors = Mutex::new(0usize);

    pool.install(|| {
        files.par_iter().for_each(|file| match registry.load(file) {
            Ok(doc) => documents.lock().unwrap().push(doc),
            Err(e) => {
                eprintln!("  Warning: failed to load {}: {}", file.display(), e);
                *load_errors.lock().unwrap() += 1;
            }
        });
    });

    let documents = documents.into_inner().unwrap();
    let load_errors = load_errors.into_inner().unwrap();
    finish_load_report(documents, load_errors)
}

fn finish_load_report(documents: Vec<Document>, load_errors: usize) -> Result<Vec<Document>> {
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
    dedup: bool,
) -> Result<(Vec<PersistedChunk>, Vec<Vec<f32>>)> {
    let mut all_chunks = Vec::new();
    let mut all_embeddings = Vec::new();

    let mut seen: HashSet<u64> = HashSet::new();
    let mut dedup_count = 0usize;
    let mut skipped_empty = 0usize;
    for doc in documents {
        if doc.content.is_empty() {
            skipped_empty += 1;
            continue;
        }
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
            if dedup {
                let mut hasher = std::hash::DefaultHasher::new();
                chunk.content.hash(&mut hasher);
                if !seen.insert(hasher.finish()) {
                    dedup_count += 1;
                    continue;
                }
            }
            all_embeddings.push(embedder.embed(&chunk.content)?);
            all_chunks.push(PersistedChunk {
                content: chunk.content.clone(),
                title: chunk.metadata.title.clone(),
                source: doc.source.clone(),
                start_secs: chunk
                    .metadata
                    .custom
                    .get("start_secs")
                    .and_then(serde_json::Value::as_f64),
                end_secs: chunk
                    .metadata
                    .custom
                    .get("end_secs")
                    .and_then(serde_json::Value::as_f64),
            });
        }
    }

    if skipped_empty > 0 {
        println!("Skipped {} empty documents", skipped_empty);
    }
    if dedup && dedup_count > 0 {
        println!("Deduplicated: removed {} duplicate chunks", dedup_count);
    }

    Ok((all_chunks, all_embeddings))
}

/// Discover files and load documents, reporting progress.
fn discover_and_load(
    path: &Path,
    recursive: bool,
    jobs: usize,
    exclude: &Option<GlobSet>,
) -> Result<(Vec<PathBuf>, HashMap<String, usize>, Vec<Document>)> {
    let registry = LoaderRegistry::new();
    let files = discover_files(path, recursive, &registry, exclude)?;

    if files.is_empty() {
        let exts = registry.supported_extensions().join(", ");
        anyhow::bail!(
            "No supported files found at: {} (supported: {})",
            path.display(),
            exts
        );
    }

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

    if jobs > 1 {
        println!("Loading with {} parallel jobs", jobs);
    }
    let documents = load_documents(&files, &registry, jobs)?;
    report_media_text_split(&documents);
    Ok((files, classification, documents))
}

/// Print how many documents have timestamp metadata vs plain text.
fn report_media_text_split(documents: &[Document]) {
    let media_count = documents
        .iter()
        .filter(|d| d.metadata.contains_key("subtitle_cues"))
        .count();
    if media_count > 0 {
        let text_count = documents.len() - media_count;
        println!(
            "  {} with timestamps, {} plain text",
            media_count, text_count
        );
    }
}

/// Create an embedder based on the selected type and return it with metadata.
fn create_embedder(
    embedder_type: EmbedderType,
    dimension: usize,
    #[allow(unused_variables)] model: SemanticModel,
    documents: &[Document],
) -> Result<(Box<dyn Embedder>, usize, String, Option<String>)> {
    match embedder_type {
        EmbedderType::Tfidf => {
            let mut embedder = TfIdfEmbedder::new(dimension);
            let doc_texts: Vec<&str> = documents.iter().map(|d| d.content.as_str()).collect();
            embedder.fit(&doc_texts);
            println!("Using TF-IDF embedder (dimension: {})", dimension);
            Ok((Box::new(embedder), dimension, "tfidf".to_string(), None))
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
                Ok((Box::new(embedder), dim, "semantic".to_string(), Some(name)))
            }
            #[cfg(not(feature = "embeddings"))]
            {
                anyhow::bail!(
                    "Semantic embeddings require the 'embeddings' feature.\n\
                     Build with: cargo build --features embeddings"
                );
            }
        }
    }
}

/// Save a persisted index to disk, optionally writing a manifest and/or SQLite export.
fn save_index(
    persisted: &PersistedIndex,
    output: &str,
    manifest: bool,
    files: &[PathBuf],
    classification: &HashMap<String, usize>,
    sqlite: bool,
) -> Result<()> {
    let output_path = Path::new(output);
    fs::create_dir_all(output_path)?;

    let index_file = output_path.join("index.json");
    let json = serde_json::to_string_pretty(persisted)?;
    fs::write(&index_file, json)?;
    println!("Index saved to: {}", index_file.display());

    if manifest {
        let manifest_data = build_index_manifest(files, classification, &persisted.chunks);
        let manifest_file = output_path.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&manifest_data)?;
        fs::write(&manifest_file, manifest_json)?;
        println!("Manifest saved to: {}", manifest_file.display());
    }

    if sqlite {
        export_sqlite(persisted, output_path)?;
    }

    Ok(())
}

/// Export a `PersistedIndex` to a `SqliteIndex` for FTS5 BM25 search.
#[cfg(feature = "sqlite")]
fn export_sqlite(persisted: &PersistedIndex, output_path: &Path) -> Result<()> {
    use std::collections::BTreeMap;

    let db_path = output_path.join("index.sqlite");
    // Remove stale DB so we get a clean export
    if db_path.exists() {
        fs::remove_file(&db_path)?;
    }
    let sqlite_index = trueno_rag::SqliteIndex::open(&db_path)
        .map_err(|e| anyhow::anyhow!("Failed to create SQLite index: {e}"))?;

    // Group chunks by source document
    let mut doc_chunks: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    let mut doc_titles: HashMap<String, Option<String>> = HashMap::new();
    for (i, pc) in persisted.chunks.iter().enumerate() {
        let doc_id = pc
            .source
            .as_deref()
            .unwrap_or("unknown")
            .to_string();
        let chunk_id = format!("{}#{}", doc_id, i);
        doc_chunks
            .entry(doc_id.clone())
            .or_default()
            .push((chunk_id, pc.content.clone()));
        doc_titles.entry(doc_id).or_insert_with(|| pc.title.clone());
    }

    let doc_count = doc_chunks.len();
    let chunk_count = persisted.chunks.len();

    for (doc_id, chunks) in &doc_chunks {
        let title = doc_titles.get(doc_id).and_then(|t| t.as_deref());
        // Concatenate all chunk content as the document-level content
        let content: String = chunks.iter().map(|(_, c)| c.as_str()).collect::<Vec<_>>().join("\n");
        sqlite_index
            .insert_document(doc_id, title, Some(doc_id.as_str()), &content, chunks, None)
            .map_err(|e| anyhow::anyhow!("Failed to insert document {doc_id}: {e}"))?;
    }

    sqlite_index
        .optimize()
        .map_err(|e| anyhow::anyhow!("Failed to optimize SQLite index: {e}"))?;

    println!(
        "SQLite index saved to: {} ({} docs, {} chunks)",
        db_path.display(),
        doc_count,
        chunk_count,
    );
    Ok(())
}

/// Stub when sqlite feature is not enabled.
#[cfg(not(feature = "sqlite"))]
fn export_sqlite(_persisted: &PersistedIndex, _output_path: &Path) -> Result<()> {
    anyhow::bail!(
        "SQLite export requires the 'sqlite' feature.\n\
         Build with: cargo build --features sqlite"
    );
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
    jobs: usize,
    manifest: bool,
    exclude_patterns: &[String],
    dedup: bool,
    sqlite: bool,
) -> Result<()> {
    let path = Path::new(path);
    if !path.exists() {
        anyhow::bail!("Path not found: {}", path.display());
    }

    let exclude = build_exclude_set(exclude_patterns)?;
    let (files, classification, documents) = discover_and_load(path, recursive, jobs, &exclude)?;
    let (embedder_box, actual_dimension, embedder_name, model_name) =
        create_embedder(embedder_type, dimension, model, &documents)?;

    let recursive_chunker = RecursiveChunker::new(chunk_size, chunk_overlap);
    let timestamp_chunker = TimestampChunker::new(60.0);
    let (all_chunks, all_embeddings) = chunk_and_embed(
        &documents,
        &*embedder_box,
        &recursive_chunker,
        &timestamp_chunker,
        chunk_strategy,
        dedup,
    )?;

    println!(
        "Indexed {} documents ({} chunks)",
        documents.len(),
        all_chunks.len()
    );

    let persisted = PersistedIndex {
        chunks: all_chunks,
        embeddings: all_embeddings,
        dimension: actual_dimension,
        embedder_type: embedder_name,
        model_name,
    };

    save_index(&persisted, output, manifest, &files, &classification, sqlite)
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

fn run_query(
    query: &str,
    index_path: &str,
    top_k: usize,
    format: &str,
    mode: &str,
    fusion: &str,
    fusion_k: Option<f32>,
    candidates: usize,
    rerank: &str,
) -> Result<()> {
    if !["dense", "sparse", "hybrid"].contains(&mode) {
        anyhow::bail!("Unknown mode: {mode} (expected dense, sparse, hybrid)");
    }

    let index_path = Path::new(index_path);
    let index_file = index_path.join("index.json");

    if !index_file.exists() {
        anyhow::bail!("Index not found at: {}", index_file.display());
    }

    let json = fs::read_to_string(&index_file)?;
    let persisted: PersistedIndex = serde_json::from_str(&json)?;

    // Fetch more candidates if reranking (reranker re-orders, so we need a wider pool)
    let retrieval_k = if rerank == "none" { top_k } else { top_k * 3 };

    let scores = match mode {
        "dense" => query_dense(query, &persisted, retrieval_k)?,
        "sparse" => query_sparse(query, &persisted, retrieval_k),
        "hybrid" => query_hybrid(query, &persisted, retrieval_k, fusion, fusion_k, candidates)?,
        _ => unreachable!(),
    };

    // Apply reranking if requested
    let scores = apply_rerank(rerank, query, &scores, &persisted.chunks, top_k)?;

    format_query_results(query, &scores, &persisted.chunks, format)
}

/// Create the correct embedder for querying based on the index's embedder_type.
///
/// If the index was built with semantic embeddings (BGE/MiniLM), returns a
/// `FastEmbedder`; otherwise returns a `TfIdfEmbedder` fit on the corpus.
fn create_query_embedder(persisted: &PersistedIndex) -> Result<Box<dyn Embedder>> {
    if persisted.embedder_type == "semantic" {
        #[cfg(feature = "embeddings")]
        {
            let model_type = match persisted.model_name.as_deref() {
                Some(name) if name.contains("bge-base") => EmbeddingModelType::BgeBaseEnV15,
                Some(name) if name.contains("bge-small") => EmbeddingModelType::BgeSmallEnV15,
                _ => EmbeddingModelType::AllMiniLmL6V2,
            };
            println!(
                "Using semantic embedder: {} (dim={})",
                model_type.model_name(),
                model_type.dimension()
            );
            let emb = FastEmbedder::new(model_type)
                .context("Failed to initialize semantic embedder")?;
            Ok(Box::new(emb))
        }
        #[cfg(not(feature = "embeddings"))]
        {
            anyhow::bail!(
                "This index uses semantic embeddings.\n\
                 Build with: cargo build --features embeddings"
            );
        }
    } else {
        let mut emb = TfIdfEmbedder::new(persisted.dimension);
        let refs: Vec<&str> = persisted
            .chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect();
        emb.fit(&refs);
        Ok(Box::new(emb))
    }
}

/// Apply reranking to scored results.
///
/// Takes `(chunk_index, score)` pairs and reranks using the specified strategy.
/// Returns `(chunk_index, rerank_score)` pairs truncated to `top_k`.
fn apply_rerank(
    rerank: &str,
    query: &str,
    scores: &[(usize, f32)],
    chunks: &[PersistedChunk],
    top_k: usize,
) -> Result<Vec<(usize, f32)>> {
    use trueno_rag::rerank::Reranker;
    use trueno_rag::retrieve::RetrievalResult;
    use trueno_rag::{Chunk, DocumentId};

    match rerank {
        "none" => Ok(scores.iter().take(top_k).copied().collect()),
        "lexical" => {
            // Convert (index, score) pairs into RetrievalResult for the Reranker trait
            let candidates: Vec<RetrievalResult> = scores
                .iter()
                .map(|(idx, score)| {
                    let pc = &chunks[*idx];
                    let mut chunk = Chunk::new(
                        DocumentId::new(),
                        pc.content.clone(),
                        0,
                        pc.content.len(),
                    );
                    // Store original index in metadata for round-tripping
                    chunk.metadata.custom.insert(
                        "_idx".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(*idx)),
                    );
                    RetrievalResult {
                        chunk,
                        dense_score: Some(*score),
                        sparse_score: None,
                        fused_score: None,
                        rerank_score: None,
                    }
                })
                .collect();

            let reranker = LexicalReranker::new();
            let reranked = reranker.rerank(query, &candidates, top_k)?;

            Ok(reranked
                .into_iter()
                .map(|rr| {
                    let idx = rr.chunk.metadata.custom.get("_idx")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let score = rr.rerank_score.unwrap_or(rr.best_score());
                    (idx, score)
                })
                .collect())
        }
        _ => anyhow::bail!("Unknown rerank strategy: {rerank} (expected none, lexical)"),
    }
}

/// Rerank `RetrievedChunk` results (used in eval retrieve path).
#[cfg(feature = "eval")]
fn rerank_retrieved_chunks(
    rerank: &str,
    query: &str,
    mut results: Vec<trueno_rag::eval::types::RetrievedChunk>,
    top_k: usize,
) -> Result<Vec<trueno_rag::eval::types::RetrievedChunk>> {
    use trueno_rag::rerank::Reranker;
    use trueno_rag::retrieve::RetrievalResult;
    use trueno_rag::{Chunk, DocumentId};

    match rerank {
        "none" => {
            results.truncate(top_k);
            Ok(results)
        }
        "lexical" => {
            let candidates: Vec<RetrievalResult> = results
                .iter()
                .enumerate()
                .map(|(i, rc)| {
                    let mut chunk = Chunk::new(
                        DocumentId::new(),
                        rc.content.clone(),
                        0,
                        rc.content.len(),
                    );
                    chunk.metadata.custom.insert(
                        "_idx".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(i)),
                    );
                    RetrievalResult {
                        chunk,
                        dense_score: Some(rc.score),
                        sparse_score: None,
                        fused_score: None,
                        rerank_score: None,
                    }
                })
                .collect();

            let reranker = LexicalReranker::new();
            let reranked = reranker.rerank(query, &candidates, top_k)?;

            Ok(reranked
                .into_iter()
                .map(|rr| {
                    let idx = rr.chunk.metadata.custom.get("_idx")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let mut rc = results[idx].clone();
                    rc.score = rr.rerank_score.unwrap_or(rr.best_score());
                    rc
                })
                .collect())
        }
        _ => anyhow::bail!("Unknown rerank strategy: {rerank} (expected none, lexical)"),
    }
}

/// Dense retrieval: TF-IDF or semantic cosine similarity.
fn query_dense(query: &str, persisted: &PersistedIndex, top_k: usize) -> Result<Vec<(usize, f32)>> {
    let embedder = create_query_embedder(persisted)?;
    let query_embedding = embedder.embed(query)?;

    let mut scores: Vec<(usize, f32)> = persisted
        .embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_similarity(&query_embedding, emb)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);
    Ok(scores)
}

/// Sparse retrieval: BM25 keyword matching.
fn query_sparse(query: &str, persisted: &PersistedIndex, top_k: usize) -> Vec<(usize, f32)> {
    use trueno_rag::index::SparseIndex;
    use trueno_rag::{BM25Index, DocumentId};

    let mut bm25 = BM25Index::new();
    let mut chunk_map: HashMap<trueno_rag::ChunkId, usize> = HashMap::new();
    for (i, pc) in persisted.chunks.iter().enumerate() {
        let chunk = Chunk::new(DocumentId::new(), pc.content.clone(), 0, pc.content.len());
        chunk_map.insert(chunk.id, i);
        bm25.add(&chunk);
    }

    let bm25_results = bm25.search(query, top_k);
    bm25_results
        .iter()
        .map(|(chunk_id, score)| (chunk_map[chunk_id], *score))
        .collect()
}

/// Hybrid retrieval: BM25 + dense (TF-IDF or semantic) with fusion.
fn query_hybrid(
    query: &str,
    persisted: &PersistedIndex,
    top_k: usize,
    fusion: &str,
    fusion_k: Option<f32>,
    candidates: usize,
) -> Result<Vec<(usize, f32)>> {
    use trueno_rag::index::VectorStoreConfig;
    use trueno_rag::retrieve::HybridRetrieverConfig;
    use trueno_rag::{BM25Index, DocumentId, HybridRetriever, VectorStore};

    let fusion_strategy = parse_fusion_strategy(fusion, fusion_k)?;

    let embedder = create_query_embedder(persisted)?;
    let dim = embedder.dimension();

    let dense_store = VectorStore::new(VectorStoreConfig {
        dimension: dim,
        ..Default::default()
    });
    let bm25 = BM25Index::new();

    let config = HybridRetrieverConfig {
        candidates_per_source: candidates,
        fusion: fusion_strategy,
        use_dense: true,
        use_sparse: true,
    };

    let mut retriever = HybridRetriever::new(dense_store, bm25, embedder).with_config(config);

    let mut chunk_meta: HashMap<trueno_rag::ChunkId, usize> = HashMap::new();
    for (i, pc) in persisted.chunks.iter().enumerate() {
        let mut chunk = Chunk::new(DocumentId::new(), pc.content.clone(), 0, pc.content.len());
        chunk.metadata.title = pc.title.clone();
        chunk.embedding = Some(persisted.embeddings[i].clone());
        chunk_meta.insert(chunk.id, i);
        retriever.index(chunk)?;
    }

    let results = retriever.retrieve(query, top_k)?;
    Ok(results
        .iter()
        .map(|rr| {
            let score = rr.fused_score.unwrap_or(rr.best_score());
            let idx = chunk_meta.get(&rr.chunk.id).copied().unwrap_or(0);
            (idx, score)
        })
        .collect())
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

#[cfg(feature = "eval")]
fn run_eval(action: EvalAction) -> Result<()> {
    match action {
        EvalAction::Sample {
            index,
            output,
            sample_size,
            seed,
        } => run_eval_sample(&index, &output, sample_size, seed),

        EvalAction::Generate {
            index,
            output,
            sample_size,
            seed,
            model,
            dry_run,
        } => run_eval_generate(&index, &output, sample_size, seed, &model, dry_run),

        EvalAction::Retrieve {
            index,
            ground_truth,
            output,
            top_k,
            mode,
            fusion,
            fusion_k,
            candidates,
            rerank,
        } => run_eval_retrieve(&index, &ground_truth, &output, top_k, &mode, &fusion, fusion_k, candidates, &rerank),

        EvalAction::Judge {
            retrieval_results,
            ground_truth: _,
            output,
            cache,
            top_k,
            model,
        } => {
            let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
            rt.block_on(run_eval_judge(
                &retrieval_results,
                &output,
                &cache,
                top_k,
                &model,
            ))
        }

        EvalAction::Metrics {
            retrieval_results,
            judgments,
            output,
        } => run_eval_metrics(&retrieval_results, &judgments, &output),

        EvalAction::Compare {
            baseline,
            candidate,
        } => run_eval_compare(&baseline, &candidate),

        EvalAction::Gate {
            results,
            min_mrr,
            min_hit5,
        } => run_eval_gate(&results, min_mrr, min_hit5),
    }
}

#[cfg(feature = "eval")]
fn run_eval_sample(
    index_path: &str,
    output_path: &str,
    sample_size: usize,
    seed: u64,
) -> Result<()> {
    use trueno_rag::eval::generate::IndexChunk;
    use trueno_rag::eval::{AnthropicClient, GroundTruthGenerator};

    let index_file = Path::new(index_path).join("index.json");
    if !index_file.exists() {
        anyhow::bail!("Index not found: {}", index_file.display());
    }

    println!("Loading index from {}...", index_file.display());
    let json = fs::read_to_string(&index_file)?;
    let persisted: PersistedIndex = serde_json::from_str(&json)?;

    let chunks: Vec<IndexChunk> = persisted
        .chunks
        .iter()
        .map(|c| IndexChunk {
            content: c.content.clone(),
            source: c.source.clone().unwrap_or_default(),
            title: c.title.clone(),
            start_secs: c.start_secs,
            end_secs: c.end_secs,
        })
        .collect();

    println!("Loaded {} chunks", chunks.len());

    // Use a dummy client — sample_chunks doesn't call the API
    let client = AnthropicClient::new("sample-only");
    let gen = GroundTruthGenerator::new(client, "none", sample_size, seed);
    let sampled = gen.sample_chunks(&chunks);

    // Write sampled chunks as JSONL
    let mut file = std::io::BufWriter::new(fs::File::create(output_path)?);
    use std::io::Write;

    #[derive(serde::Serialize)]
    struct SampledChunkOutput {
        content: String,
        source: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        start_secs: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        end_secs: Option<f64>,
        course: String,
        domain: String,
    }

    for s in &sampled {
        let entry = SampledChunkOutput {
            content: s.content.clone(),
            source: s.source.clone(),
            start_secs: s.start_secs,
            end_secs: s.end_secs,
            course: s.course.clone(),
            domain: s.domain.clone(),
        };
        serde_json::to_writer(&mut file, &entry)?;
        writeln!(file)?;
    }

    println!(
        "\nSampled {} chunks saved to: {output_path}",
        sampled.len()
    );
    Ok(())
}

#[cfg(feature = "eval")]
fn run_eval_generate(
    index_path: &str,
    output_path: &str,
    sample_size: usize,
    seed: u64,
    model: &str,
    dry_run: bool,
) -> Result<()> {
    use trueno_rag::eval::{generate::IndexChunk, AnthropicClient, GroundTruthGenerator};

    let index_file = Path::new(index_path).join("index.json");
    if !index_file.exists() {
        anyhow::bail!("Index not found: {}", index_file.display());
    }

    println!("Loading index from {}...", index_file.display());
    let json = fs::read_to_string(&index_file)?;
    let persisted: PersistedIndex = serde_json::from_str(&json)?;

    let chunks: Vec<IndexChunk> = persisted
        .chunks
        .iter()
        .map(|c| IndexChunk {
            content: c.content.clone(),
            source: c.source.clone().unwrap_or_default(),
            title: c.title.clone(),
            start_secs: c.start_secs,
            end_secs: c.end_secs,
        })
        .collect();

    println!("Loaded {} chunks", chunks.len());

    if dry_run {
        let client = AnthropicClient::new("dry-run");
        let gen = GroundTruthGenerator::new(client, model, sample_size, seed);
        let sampled = gen.sample_chunks(&chunks);
        println!("\nDry run: would generate {} questions", sampled.len());
        for s in sampled.iter().take(10) {
            println!(
                "  [{}] {}: {}...",
                s.domain,
                s.course,
                &s.content[..s.content.len().min(80)]
            );
        }
        return Ok(());
    }

    let client = AnthropicClient::from_env()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let gen = GroundTruthGenerator::new(client, model, sample_size, seed);

    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    let results = rt.block_on(gen.generate(&chunks))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Write JSONL output
    let mut file = std::io::BufWriter::new(fs::File::create(output_path)?);
    use std::io::Write;
    for entry in &results {
        serde_json::to_writer(&mut file, entry)?;
        writeln!(file)?;
    }

    println!("\nGround truth saved to: {output_path} ({} entries)", results.len());
    Ok(())
}

fn parse_fusion_strategy(fusion: &str, fusion_k: Option<f32>) -> Result<FusionStrategy> {
    match fusion {
        "rrf" => Ok(FusionStrategy::RRF { k: fusion_k.unwrap_or(60.0) }),
        "linear" => Ok(FusionStrategy::Linear { dense_weight: fusion_k.unwrap_or(0.5) }),
        "dbsf" => Ok(FusionStrategy::DBSF),
        other => anyhow::bail!("Unknown fusion strategy: {other} (expected rrf, linear, dbsf)"),
    }
}

#[cfg(feature = "eval")]
fn run_eval_retrieve(
    index_path: &str,
    ground_truth_path: &str,
    output_path: &str,
    top_k: usize,
    mode: &str,
    fusion: &str,
    fusion_k: Option<f32>,
    candidates: usize,
    rerank: &str,
) -> Result<()> {
    use trueno_rag::eval::types::{GroundTruthEntry, RetrievalResultEntry, RetrievedChunk};
    use trueno_rag::index::{SparseIndex, VectorStoreConfig};
    use trueno_rag::retrieve::HybridRetrieverConfig;
    use trueno_rag::{BM25Index, DocumentId, HybridRetriever, VectorStore};

    // Validate mode
    if !["dense", "sparse", "hybrid"].contains(&mode) {
        anyhow::bail!("Unknown mode: {mode} (expected dense, sparse, hybrid)");
    }

    let index_file = Path::new(index_path).join("index.json");
    if !index_file.exists() {
        anyhow::bail!("Index not found: {}", index_file.display());
    }

    // Load ground truth
    let gt_text = fs::read_to_string(ground_truth_path)
        .with_context(|| format!("Failed to read {ground_truth_path}"))?;
    let queries: Vec<GroundTruthEntry> = gt_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .context("Failed to parse ground truth JSONL")?;

    println!("Loaded {} queries from {}", queries.len(), ground_truth_path);
    println!("Mode: {mode} | Fusion: {fusion} | Candidates: {candidates} | Top-k: {top_k} | Rerank: {rerank}");

    // Fetch more candidates when reranking to give the reranker a wider pool
    let retrieval_k = if rerank == "none" { top_k } else { top_k * 3 };

    // Load index
    println!("Loading index from {}...", index_file.display());
    let json = fs::read_to_string(&index_file)?;
    let persisted: PersistedIndex = serde_json::from_str(&json)?;
    println!("Index: {} chunks, dim={}", persisted.chunks.len(), persisted.dimension);

    // Build embedder (auto-detects semantic vs TF-IDF from index metadata)
    let embedder = create_query_embedder(&persisted)?;

    // Convert PersistedChunks to Chunks with embeddings for hybrid/sparse modes
    let build_chunks = |with_embeddings: bool| -> Vec<Chunk> {
        persisted.chunks.iter().enumerate().map(|(i, pc)| {
            let mut chunk = Chunk::new(DocumentId::new(), pc.content.clone(), 0, pc.content.len());
            chunk.metadata.title = pc.title.clone();
            chunk.metadata.custom.insert(
                "source".to_string(),
                serde_json::Value::String(pc.source.clone().unwrap_or_default()),
            );
            if with_embeddings {
                chunk.embedding = Some(persisted.embeddings[i].clone());
            }
            chunk
        }).collect()
    };

    // Run queries
    let mut output_file = std::io::BufWriter::new(fs::File::create(output_path)?);
    use std::io::Write;

    match mode {
        "dense" => {
            // Dense cosine similarity (TF-IDF or semantic, auto-detected)
            for (i, entry) in queries.iter().enumerate() {
                print!(
                    "[{}/{}] {}...",
                    i + 1, queries.len(),
                    &entry.query[..entry.query.len().min(60)]
                );

                let start = std::time::Instant::now();
                let query_embedding = embedder.embed(&entry.query)?;

                let mut scores: Vec<(usize, f32)> = persisted
                    .embeddings.iter().enumerate()
                    .map(|(idx, emb)| (idx, cosine_similarity(&query_embedding, emb)))
                    .collect();

                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scores.truncate(retrieval_k);
                let latency = start.elapsed().as_secs_f64();

                let results: Vec<RetrievedChunk> = scores.iter().map(|(idx, score)| {
                    let chunk = &persisted.chunks[*idx];
                    RetrievedChunk {
                        content: chunk.content.clone(),
                        source: chunk.source.clone(),
                        score: *score,
                        title: chunk.title.clone(),
                        start_secs: chunk.start_secs,
                        end_secs: chunk.end_secs,
                    }
                }).collect();

                let results = rerank_retrieved_chunks(rerank, &entry.query, results, top_k)?;

                println!(" {} results ({:.2}s)", results.len(), latency);
                let result_entry = RetrievalResultEntry {
                    query: entry.query.clone(),
                    domain: entry.domain.clone(),
                    course: entry.course.clone(),
                    results,
                    latency_s: latency,
                };
                serde_json::to_writer(&mut output_file, &result_entry)?;
                writeln!(output_file)?;
            }
        }

        "sparse" => {
            // BM25-only retrieval
            println!("Building BM25 index from {} chunks...", persisted.chunks.len());
            let start_build = std::time::Instant::now();
            let mut bm25 = BM25Index::new();
            let chunks = build_chunks(false);
            // We need a mapping from ChunkId back to index for metadata
            let mut chunk_map: HashMap<trueno_rag::ChunkId, usize> = HashMap::new();
            for (i, chunk) in chunks.iter().enumerate() {
                chunk_map.insert(chunk.id, i);
                bm25.add(chunk);
            }
            println!("BM25 index built in {:.2}s", start_build.elapsed().as_secs_f64());

            for (i, entry) in queries.iter().enumerate() {
                print!(
                    "[{}/{}] {}...",
                    i + 1, queries.len(),
                    &entry.query[..entry.query.len().min(60)]
                );

                let start = std::time::Instant::now();
                let bm25_results = bm25.search(&entry.query, retrieval_k);
                let latency = start.elapsed().as_secs_f64();

                let results: Vec<RetrievedChunk> = bm25_results.iter().map(|(chunk_id, score)| {
                    let idx = chunk_map[chunk_id];
                    let pc = &persisted.chunks[idx];
                    RetrievedChunk {
                        content: pc.content.clone(),
                        source: pc.source.clone(),
                        score: *score,
                        title: pc.title.clone(),
                        start_secs: pc.start_secs,
                        end_secs: pc.end_secs,
                    }
                }).collect();

                let results = rerank_retrieved_chunks(rerank, &entry.query, results, top_k)?;

                println!(" {} results ({:.2}s)", results.len(), latency);
                let result_entry = RetrievalResultEntry {
                    query: entry.query.clone(),
                    domain: entry.domain.clone(),
                    course: entry.course.clone(),
                    results,
                    latency_s: latency,
                };
                serde_json::to_writer(&mut output_file, &result_entry)?;
                writeln!(output_file)?;
            }
        }

        "hybrid" => {
            // Hybrid retrieval: BM25 + dense (TF-IDF or semantic) with fusion
            let fusion_strategy = parse_fusion_strategy(fusion, fusion_k)?;
            let dim = embedder.dimension();
            println!("Building hybrid retriever (BM25 + dense dim={dim}, fusion={fusion})...");
            let start_build = std::time::Instant::now();

            let dense_store = VectorStore::new(VectorStoreConfig {
                dimension: dim,
                ..Default::default()
            });
            let bm25 = BM25Index::new();

            let config = HybridRetrieverConfig {
                candidates_per_source: candidates,
                fusion: fusion_strategy,
                use_dense: true,
                use_sparse: true,
            };

            let mut retriever = HybridRetriever::new(dense_store, bm25, embedder)
                .with_config(config);

            // Index all chunks (populates both BM25 and VectorStore)
            let chunks = build_chunks(true);
            let n_chunks = chunks.len();
            // We need a mapping from ChunkId to persisted index for metadata
            let mut chunk_meta: HashMap<trueno_rag::ChunkId, usize> = HashMap::new();
            for (i, chunk) in chunks.into_iter().enumerate() {
                chunk_meta.insert(chunk.id, i);
                retriever.index(chunk)?;
            }
            println!("Hybrid retriever built: {} chunks in {:.2}s", n_chunks, start_build.elapsed().as_secs_f64());

            for (i, entry) in queries.iter().enumerate() {
                print!(
                    "[{}/{}] {}...",
                    i + 1, queries.len(),
                    &entry.query[..entry.query.len().min(60)]
                );

                let start = std::time::Instant::now();
                let retrieval_results = retriever.retrieve(&entry.query, retrieval_k)?;
                let latency = start.elapsed().as_secs_f64();

                let results: Vec<RetrievedChunk> = retrieval_results.iter().map(|rr| {
                    let score = rr.fused_score.unwrap_or(rr.best_score());
                    // Look up metadata from persisted index via chunk_meta
                    if let Some(&idx) = chunk_meta.get(&rr.chunk.id) {
                        let pc = &persisted.chunks[idx];
                        RetrievedChunk {
                            content: pc.content.clone(),
                            source: pc.source.clone(),
                            score,
                            title: pc.title.clone(),
                            start_secs: pc.start_secs,
                            end_secs: pc.end_secs,
                        }
                    } else {
                        RetrievedChunk {
                            content: rr.chunk.content.clone(),
                            source: None,
                            score,
                            title: None,
                            start_secs: None,
                            end_secs: None,
                        }
                    }
                }).collect();

                let results = rerank_retrieved_chunks(rerank, &entry.query, results, top_k)?;

                println!(" {} results ({:.2}s)", results.len(), latency);
                let result_entry = RetrievalResultEntry {
                    query: entry.query.clone(),
                    domain: entry.domain.clone(),
                    course: entry.course.clone(),
                    results,
                    latency_s: latency,
                };
                serde_json::to_writer(&mut output_file, &result_entry)?;
                writeln!(output_file)?;
            }
        }

        _ => unreachable!(),
    }

    println!("\nRetrieval results saved to: {output_path}");
    Ok(())
}

#[cfg(feature = "eval")]
async fn run_eval_judge(
    retrieval_results_path: &str,
    output_path: &str,
    cache_path: &str,
    top_k: usize,
    model: &str,
) -> Result<()> {
    use trueno_rag::eval::{
        types::{JudgeCache, RetrievalResultEntry},
        AnthropicClient, RelevanceJudge,
    };

    // Load retrieval results
    let text = fs::read_to_string(retrieval_results_path)
        .with_context(|| format!("Failed to read {retrieval_results_path}"))?;
    let results: Vec<RetrievalResultEntry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .context("Failed to parse retrieval results JSONL")?;

    println!("Loaded {} retrieval results", results.len());

    // Load cache
    let cache = JudgeCache::load(Path::new(cache_path));
    println!("Cache: {} entries loaded from {}", cache.entries.len(), cache_path);

    let client = AnthropicClient::from_env()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut judge = RelevanceJudge::new(client, model, cache);

    let eval_output = judge.evaluate(&results, top_k).await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Save cache
    judge.cache().save(Path::new(cache_path))
        .context("Failed to save judge cache")?;
    println!("Cache saved: {} entries to {}", judge.cache().entries.len(), cache_path);

    // Save results
    let json = serde_json::to_string_pretty(&eval_output)?;
    fs::write(output_path, json)?;
    println!("Results saved to: {output_path}");

    Ok(())
}

#[cfg(feature = "eval")]
fn run_eval_metrics(
    retrieval_results_path: &str,
    judgments_path: &str,
    output_path: &str,
) -> Result<()> {
    use trueno_rag::eval::metrics::{compute_metrics_from_judgments, format_metrics_summary};
    use trueno_rag::eval::types::{JudgmentEntry, RetrievalResultEntry};

    // Load retrieval results
    let text = fs::read_to_string(retrieval_results_path)
        .with_context(|| format!("Failed to read {retrieval_results_path}"))?;
    let results: Vec<RetrievalResultEntry> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .context("Failed to parse retrieval results JSONL")?;

    // Load judgments
    let jtext = fs::read_to_string(judgments_path)
        .with_context(|| format!("Failed to read {judgments_path}"))?;
    let judgments: Vec<JudgmentEntry> = jtext
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .context("Failed to parse judgments JSONL")?;

    println!(
        "Loaded {} retrieval results, {} judgments",
        results.len(),
        judgments.len()
    );

    let eval_output = compute_metrics_from_judgments(&results, &judgments);

    println!(
        "\n{}",
        format_metrics_summary(&eval_output.aggregate, &eval_output.by_domain)
    );

    let json = serde_json::to_string_pretty(&eval_output)?;
    fs::write(output_path, json)?;
    println!("Results saved to: {output_path}");

    Ok(())
}

#[cfg(feature = "eval")]
fn run_eval_compare(baseline_path: &str, candidate_path: &str) -> Result<()> {
    use trueno_rag::eval::{judge::compare_results, types::EvalOutput};

    let baseline: EvalOutput = serde_json::from_str(
        &fs::read_to_string(baseline_path)
            .with_context(|| format!("Failed to read {baseline_path}"))?,
    )?;
    let candidate: EvalOutput = serde_json::from_str(
        &fs::read_to_string(candidate_path)
            .with_context(|| format!("Failed to read {candidate_path}"))?,
    )?;

    println!("{}", compare_results(&baseline, &candidate));
    Ok(())
}

#[cfg(feature = "eval")]
fn run_eval_gate(results_path: &str, min_mrr: f64, min_hit5: f64) -> Result<()> {
    use trueno_rag::eval::{judge::check_gate, types::EvalOutput};

    let output: EvalOutput = serde_json::from_str(
        &fs::read_to_string(results_path)
            .with_context(|| format!("Failed to read {results_path}"))?,
    )?;

    match check_gate(&output, min_mrr, min_hit5) {
        Ok(()) => {
            println!("Regression gate PASSED (MRR={:.4}, Hit@5={:.4})",
                output.aggregate.mrr, output.aggregate.hit_rate_5);
            Ok(())
        }
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
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
        let files = discover_files(&file, false, &registry, &None).unwrap();
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
        let files = discover_files(&dir, false, &registry, &None).unwrap();
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
        let files = discover_files(&dir, true, &registry, &None).unwrap();
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
        let result = discover_files(&file, false, &registry, &None);
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

    #[test]
    fn test_build_exclude_set_empty() {
        let result = build_exclude_set(&[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_build_exclude_set_valid() {
        let patterns = vec!["*/RAW".to_string(), "*/RAW/*".to_string()];
        let result = build_exclude_set(&patterns).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_build_exclude_set_invalid() {
        let patterns = vec!["[invalid".to_string()];
        let result = build_exclude_set(&patterns);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_excluded_matches() {
        let patterns = vec!["*/RAW".to_string(), "*/RAW/*".to_string()];
        let exclude = build_exclude_set(&patterns).unwrap();
        assert!(is_excluded(Path::new("/data/courses/aws/RAW"), &exclude));
        assert!(is_excluded(
            Path::new("/data/courses/aws/RAW/video.mp4"),
            &exclude
        ));
        assert!(!is_excluded(
            Path::new("/data/courses/aws/build/video.srt"),
            &exclude
        ));
    }

    #[test]
    fn test_is_excluded_none() {
        assert!(!is_excluded(Path::new("/any/path"), &None));
    }

    #[test]
    fn test_discover_files_with_exclude() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_exclude");
        let raw = dir.join("RAW");
        let build = dir.join("build");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&raw);
        let _ = fs::create_dir_all(&build);
        fs::write(dir.join("keep.txt"), "keep").unwrap();
        fs::write(raw.join("skip.txt"), "skip").unwrap();
        fs::write(build.join("also_keep.txt"), "keep2").unwrap();

        let registry = LoaderRegistry::new();
        let exclude = build_exclude_set(&["*/RAW".to_string(), "*/RAW/*".to_string()]).unwrap();
        let files = discover_files(&dir, true, &registry, &exclude).unwrap();

        // Should have keep.txt and build/also_keep.txt but NOT RAW/skip.txt
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| !f.to_string_lossy().contains("RAW")));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_discover_media_files_with_exclude() {
        let dir = std::env::temp_dir().join("trueno_rag_cli_test_media_exclude");
        let raw = dir.join("RAW");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::create_dir_all(&raw);
        fs::write(dir.join("keep.mp4"), "fake media").unwrap();
        fs::write(raw.join("skip.mp4"), "fake media").unwrap();

        let exclude = build_exclude_set(&["*/RAW".to_string(), "*/RAW/*".to_string()]).unwrap();
        let files = discover_media_files(&dir, true, &exclude).unwrap();

        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().contains("keep.mp4"));

        let _ = fs::remove_dir_all(&dir);
    }
}
