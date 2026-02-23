//! Evaluation framework for RAG retrieval quality (PMAT-015)
//!
//! World-class RAG evaluation using LLM-as-judge on actual chunk content
//! and synthetic ground truth generated from the corpus itself.
//!
//! # Architecture
//!
//! Split pipeline (local-first):
//! - **Retrieval** runs on the indexing machine (no API keys needed)
//! - **Generation + Judging** runs locally where `ANTHROPIC_API_KEY` exists
//!
//! # Subcommands
//!
//! - `eval generate` — Sample chunks, generate questions via Claude API
//! - `eval retrieve` — Run queries against index, dump results to JSONL
//! - `eval judge` — LLM-as-judge relevance scoring + IR metrics

pub mod client;
pub mod domain;
pub mod generate;
pub mod judge;
pub mod types;

pub use client::AnthropicClient;
pub use domain::classify_domain;
pub use generate::GroundTruthGenerator;
pub use judge::RelevanceJudge;
pub use types::{
    EvalConfig, GroundTruthEntry, JudgeCache, JudgeCacheEntry, JudgeVerdict, RetrievalResultEntry,
};
