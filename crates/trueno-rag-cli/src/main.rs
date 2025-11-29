//! Trueno-RAG CLI
//!
//! Command-line interface for the Trueno-RAG pipeline.

use anyhow::Result;
use clap::{Parser, Subcommand};
use trueno_rag::{
    chunk::RecursiveChunker,
    embed::MockEmbedder,
    fusion::FusionStrategy,
    pipeline::RagPipelineBuilder,
    rerank::LexicalReranker,
    Document,
};

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

        /// Chunk size
        #[arg(long, default_value = "512")]
        chunk_size: usize,
    },

    /// Query the RAG pipeline
    Query {
        /// Query string
        query: String,

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

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Demo { query, top_k } => run_demo(&query, top_k)?,
        Commands::Index { path, chunk_size } => {
            println!("Indexing from: {} (chunk_size: {})", path, chunk_size);
            println!("Note: Full indexing requires document loader implementation");
        }
        Commands::Query {
            query,
            top_k,
            format,
        } => {
            println!("Query: {} (top_k: {}, format: {})", query, top_k, format);
            println!("Note: Requires indexed documents");
        }
        Commands::Info => {
            println!("Trueno-RAG Pipeline");
            println!("==================");
            println!("Version: {}", env!("CARGO_PKG_VERSION"));
            println!();
            println!("Components:");
            println!("  - Chunkers: Recursive, Fixed, Sentence, Paragraph, Semantic, Structural");
            println!("  - Embedders: Mock, TF-IDF (custom implementations supported)");
            println!("  - Fusion: RRF, Linear, DBSF, Convex, Union, Intersection");
            println!("  - Rerankers: Lexical, CrossEncoder (mock), Composite");
        }
    }

    Ok(())
}

fn run_demo(query: &str, top_k: usize) -> Result<()> {
    println!("=== Trueno-RAG Demo ===\n");

    // Build pipeline
    let mut pipeline = RagPipelineBuilder::new()
        .chunker(RecursiveChunker::new(256, 32))
        .embedder(MockEmbedder::new(128))
        .reranker(LexicalReranker::new())
        .fusion(FusionStrategy::RRF { k: 60.0 })
        .max_context_tokens(2000)
        .build()?;

    // Sample documents
    let docs = vec![
        Document::new(
            "Machine learning is a subset of artificial intelligence that enables \
             systems to learn and improve from experience without being explicitly programmed.",
        )
        .with_title("Machine Learning Basics"),
        Document::new(
            "Deep learning uses neural networks with many layers to learn representations \
             of data. It has achieved breakthrough results in image and speech recognition.",
        )
        .with_title("Deep Learning Overview"),
        Document::new(
            "Natural language processing enables computers to understand, interpret, \
             and generate human language in a valuable way.",
        )
        .with_title("NLP Introduction"),
        Document::new(
            "Retrieval-Augmented Generation combines retrieval systems with generative \
             models to produce more accurate and grounded responses.",
        )
        .with_title("RAG Systems"),
    ];

    // Index
    let chunk_count = pipeline.index_documents(&docs)?;
    println!("Indexed {} documents ({} chunks)\n", docs.len(), chunk_count);

    // Query
    println!("Query: \"{}\"\n", query);

    let (results, context) = pipeline.query_with_context(query, top_k)?;

    println!("Results ({}):", results.len());
    println!("{}", "-".repeat(50));

    for (i, result) in results.iter().enumerate() {
        let title = result
            .chunk
            .metadata
            .title
            .as_deref()
            .unwrap_or("Untitled");
        println!(
            "{}. [Score: {:.3}] {}",
            i + 1,
            result.best_score(),
            title
        );
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
