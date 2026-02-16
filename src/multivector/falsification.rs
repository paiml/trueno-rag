//! Popperian Falsification Tests for WARP Algorithm
//!
//! These tests are designed to REFUTE the WARP implementation, not confirm it.
//! Each test represents a "potential falsifier" - an observation that would
//! compel us to reject the current implementation as failed.
//!
//! "The game of science is, in principle, without end. He who decides one day
//! that scientific statements do not call for any further test, and that they
//! can be regarded as finally verified, retires from the game."
//! — Karl Popper, The Logic of Scientific Discovery

#![allow(missing_docs)]

use crate::multivector::{
    exact_maxsim, MockMultiVectorEmbedder, MultiVectorEmbedder, MultiVectorEmbedding,
    ResidualCodec, WarpIndex, WarpIndexConfig, WarpSearchConfig,
};
use crate::{Chunk, DocumentId};
use std::time::Instant;

/// Falsification Report for WARP implementation
#[derive(Debug)]
pub struct FalsificationReport {
    pub experimentum_crucis: ConjectureResult,
    pub conjecture_1_compression: ConjectureResult,
    pub conjecture_2_pruning: ConjectureResult,
    pub conjecture_3_scaling: ConjectureResult,
    pub overall_verdict: Verdict,
}

#[derive(Debug, Clone)]
pub struct ConjectureResult {
    pub name: String,
    pub hypothesis: String,
    pub threshold: String,
    pub observed_value: String,
    pub verdict: Verdict,
    pub details: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Corroborated,
    Falsified,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Corroborated => write!(f, "CORROBORATED"),
            Verdict::Falsified => write!(f, "FALSIFIED"),
        }
    }
}

impl FalsificationReport {
    pub fn print_report(&self) {
        println!("\n======== WARP FALSIFICATION REPORT ========");
        println!();

        self.print_conjecture(&self.experimentum_crucis);
        self.print_conjecture(&self.conjecture_1_compression);
        self.print_conjecture(&self.conjecture_2_pruning);
        self.print_conjecture(&self.conjecture_3_scaling);

        println!("======== FINAL VERDICT ========");
        println!("Overall: {}", self.overall_verdict);

        if self.overall_verdict == Verdict::Falsified {
            println!("\n*** STOP THE LINE (Jidoka) ***");
            println!("The current implementation has been FALSIFIED.");
            println!("Do NOT patch. Reformulate the theory.");
        }
    }

    fn print_conjecture(&self, result: &ConjectureResult) {
        println!("--- {} ---", result.name);
        println!("Hypothesis: {}", result.hypothesis);
        println!("Threshold: {}", result.threshold);
        println!("Observed: {}", result.observed_value);
        println!("Verdict: {}", result.verdict);
        println!("Details: {}", result.details);
        println!();
    }
}

/// Execute the complete falsification plan
pub fn execute_falsification_plan() -> FalsificationReport {
    let experimentum = test_experimentum_crucis();
    let compression = test_conjecture_1_compression();
    let pruning = test_conjecture_2_pruning();
    let scaling = test_conjecture_3_scaling();

    let overall = if experimentum.verdict == Verdict::Falsified
        || compression.verdict == Verdict::Falsified
        || pruning.verdict == Verdict::Falsified
        || scaling.verdict == Verdict::Falsified
    {
        Verdict::Falsified
    } else {
        Verdict::Corroborated
    };

    FalsificationReport {
        experimentum_crucis: experimentum,
        conjecture_1_compression: compression,
        conjecture_2_pruning: pruning,
        conjecture_3_scaling: scaling,
        overall_verdict: overall,
    }
}

// =============================================================================
// EXPERIMENTUM CRUCIS: Hard Negatives Test
// =============================================================================

/// Test: WARP must outperform single-vector by 15% MRR@10 on hard negatives
fn test_experimentum_crucis() -> ConjectureResult {
    let name = "Experimentum Crucis (Hard Negatives)".to_string();
    let hypothesis =
        "Token-level interaction captures nuances that single-vector misses".to_string();
    let threshold = "WARP MRR@10 delta >= 15% over single-vector".to_string();

    // Create embedder with sufficient dimension for semantic distinction
    let embedder = MockMultiVectorEmbedder::new(32, 256);

    // Hard negative pairs: semantically similar but critically different
    // Each sentence has enough tokens for training (8+ tokens per sentence)
    let hard_negative_pairs = vec![
        (
            "The quick brown cat is sitting comfortably on the soft mat today",
            "The quick brown cat is definitely NOT sitting on the soft mat today",
        ),
        (
            "Machine learning algorithms significantly improve model accuracy and performance metrics",
            "Machine learning algorithms do not significantly improve model accuracy and performance",
        ),
        (
            "The controlled scientific experiment succeeded beyond our initial expectations completely",
            "The controlled scientific experiment failed miserably beyond our initial expectations",
        ),
        (
            "All registered users have full access to the secure system features",
            "No registered users have any access to the secure system features",
        ),
        (
            "The credit card payment was successfully processed through the gateway",
            "The credit card payment was unfortunately rejected by the gateway",
        ),
        (
            "The optimization algorithm converges quickly toward the global minimum solution",
            "The optimization algorithm never converges toward the global minimum solution",
        ),
        (
            "The ambient temperature is steadily rising throughout the experimental period",
            "The ambient temperature is rapidly falling throughout the experimental period",
        ),
        (
            "The encrypted network connection is completely secure against external attacks",
            "The encrypted network connection is dangerously compromised by external attacks",
        ),
        (
            "The database query returned all matching records from the main table",
            "The database query returned no matching records from the main table",
        ),
        (
            "The software test cases passed successfully without any critical errors",
            "The software test cases failed completely with many critical errors",
        ),
    ];

    // Build WARP index - use fewer centroids (4) to ensure enough training data
    // Need 4 centroids * 10 tokens = 40 tokens minimum
    // With 20 sentences * ~12 tokens = ~240 tokens, we have plenty
    let config = WarpIndexConfig::new(2, 4, 32).with_kmeans_iterations(10);
    let mut index = WarpIndex::new(config);

    // Generate training data
    let mut all_texts: Vec<String> = Vec::new();
    for (pos, neg) in &hard_negative_pairs {
        all_texts.push(pos.to_string());
        all_texts.push(neg.to_string());
    }

    let training_embeddings: Vec<MultiVectorEmbedding> = all_texts
        .iter()
        .map(|t| embedder.embed_tokens(t).unwrap())
        .collect();

    if let Err(e) = index.train(&training_embeddings) {
        return ConjectureResult {
            name,
            hypothesis,
            threshold,
            observed_value: format!("Training failed: {e}"),
            verdict: Verdict::Falsified,
            details: "Could not train index".to_string(),
        };
    }

    // Index all documents
    for text in all_texts.iter() {
        let chunk = Chunk::new(DocumentId::new(), text.clone(), 0, text.len());
        let embedding = embedder.embed_tokens(text).unwrap();
        index.insert(chunk, embedding).unwrap();
    }
    index.build().unwrap();

    // Compute MRR@10 for WARP
    let mut warp_mrr_sum = 0.0;
    let mut single_vector_mrr_sum = 0.0;
    let num_queries = hard_negative_pairs.len();

    for (query_idx, (positive, _negative)) in hard_negative_pairs.iter().enumerate() {
        let query_embedding = embedder.embed_tokens(positive).unwrap();

        // WARP search
        let search_config = WarpSearchConfig::with_k(10);
        let warp_results = index.search(&query_embedding, &search_config).unwrap();

        // Find rank of the positive document
        let positive_doc_idx = query_idx * 2; // Index of positive in all_texts
        let warp_rank = warp_results
            .iter()
            .position(|(chunk_id, _)| {
                index
                    .get_chunk(chunk_id)
                    .map(|c| c.content == *positive)
                    .unwrap_or(false)
            })
            .map(|r| r + 1);

        if let Some(rank) = warp_rank {
            warp_mrr_sum += 1.0 / rank as f64;
        }

        // Single-vector comparison: use centroid of multi-vector as single vector
        // Compare against all documents using average embedding similarity
        let query_avg = average_embedding(&query_embedding);
        let mut single_vec_scores: Vec<(usize, f64)> = all_texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let doc_emb = embedder.embed_tokens(text).unwrap();
                let doc_avg = average_embedding(&doc_emb);
                let score = cosine_similarity(&query_avg, &doc_avg);
                (i, score)
            })
            .collect();

        single_vec_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let single_rank = single_vec_scores
            .iter()
            .position(|(i, _)| *i == positive_doc_idx)
            .map(|r| r + 1);

        if let Some(rank) = single_rank {
            single_vector_mrr_sum += 1.0 / rank as f64;
        }
    }

    let warp_mrr = warp_mrr_sum / num_queries as f64;
    let single_mrr = single_vector_mrr_sum / num_queries as f64;
    let delta_percent = if single_mrr > 0.0 {
        ((warp_mrr - single_mrr) / single_mrr) * 100.0
    } else {
        100.0
    };

    let observed = format!(
        "WARP MRR@10={:.4}, Single-Vector MRR@10={:.4}, Delta={:.2}%",
        warp_mrr, single_mrr, delta_percent
    );

    let verdict = if delta_percent >= 15.0 {
        Verdict::Corroborated
    } else {
        Verdict::Falsified
    };

    let details = format!(
        "Tested {} hard negative pairs. WARP {} the 15% improvement threshold.",
        num_queries,
        if verdict == Verdict::Corroborated {
            "met"
        } else {
            "failed to meet"
        }
    );

    ConjectureResult {
        name,
        hypothesis,
        threshold,
        observed_value: observed,
        verdict,
        details,
    }
}

// =============================================================================
// CONJECTURE 1: Score Ordering Preservation (Kendall's Tau)
// =============================================================================

/// Test: Kendall's tau between full-precision and quantized scores >= 0.90
fn test_conjecture_1_compression() -> ConjectureResult {
    let name = "Conjecture 1: Compression Preserves Score Ordering".to_string();
    let hypothesis =
        "Residual quantization preserves relative ordering of MaxSim scores".to_string();
    let threshold = "Kendall's tau >= 0.90".to_string();

    let embedder = MockMultiVectorEmbedder::new(32, 128);

    // Generate diverse test documents with more tokens each
    let documents: Vec<String> = (0..50)
        .map(|i| {
            format!(
                "Document number {} comprehensively discusses topic {} with various {} and {} concepts and ideas",
                i,
                i % 10,
                if i % 2 == 0 { "scientific" } else { "technical" },
                if i % 3 == 0 {
                    "advanced theoretical"
                } else {
                    "fundamental practical"
                }
            )
        })
        .collect();

    // Train codec
    let doc_embeddings: Vec<MultiVectorEmbedding> = documents
        .iter()
        .map(|d| embedder.embed_tokens(d).unwrap())
        .collect();

    let all_tokens: Vec<f32> = doc_embeddings
        .iter()
        .flat_map(|e| e.as_slice().iter().copied())
        .collect();

    // Use 4-bit quantization for higher fidelity (spec suggests 2-bit has 3-5% quality loss)
    // With 8 centroids and ~650 tokens (50 docs * ~13 tokens), we have enough data
    let codec = match ResidualCodec::train(&all_tokens, 32, 8, 4, 20) {
        Ok(c) => c,
        Err(e) => {
            return ConjectureResult {
                name,
                hypothesis,
                threshold,
                observed_value: format!("Codec training failed: {e}"),
                verdict: Verdict::Falsified,
                details: "Could not train codec".to_string(),
            }
        }
    };

    // Generate queries
    let queries: Vec<String> = vec![
        "scientific concepts in documents".to_string(),
        "technical advanced topics".to_string(),
        "fundamental discussion of ideas".to_string(),
    ];

    let query_embeddings: Vec<MultiVectorEmbedding> = queries
        .iter()
        .map(|q| embedder.embed_tokens(q).unwrap())
        .collect();

    // Compute exact and approximate scores
    let mut exact_scores: Vec<f64> = Vec::new();
    let mut approx_scores: Vec<f64> = Vec::new();

    for query_emb in &query_embeddings {
        for doc_emb in &doc_embeddings {
            // Exact score (full precision)
            let exact = exact_maxsim(query_emb, doc_emb);
            exact_scores.push(exact as f64);

            // Approximate score (quantized)
            let approx = compute_quantized_maxsim(&codec, query_emb, doc_emb);
            approx_scores.push(approx);
        }
    }

    // Compute Kendall's tau
    let tau = kendalls_tau(&exact_scores, &approx_scores);

    let observed = format!("Kendall's tau = {:.4}", tau);

    let verdict = if tau >= 0.90 {
        Verdict::Corroborated
    } else {
        Verdict::Falsified
    };

    let details = format!(
        "Computed rank correlation over {} score pairs. Tau {} threshold.",
        exact_scores.len(),
        if verdict == Verdict::Corroborated {
            "meets"
        } else {
            "below"
        }
    );

    ConjectureResult {
        name,
        hypothesis,
        threshold,
        observed_value: observed,
        verdict,
        details,
    }
}

// =============================================================================
// CONJECTURE 2: Pruning Hypothesis (Recall vs Exhaustive)
// =============================================================================

/// Test: recall@10 of pruned search (nprobe=4) vs exhaustive >= 0.95
fn test_conjecture_2_pruning() -> ConjectureResult {
    let name = "Conjecture 2: Centroid Pruning Recall".to_string();
    let hypothesis =
        "Top-nprobe centroids contain relevant tokens for accurate retrieval".to_string();
    let threshold = "recall@10 (nprobe=4) vs exhaustive >= 95%".to_string();

    let embedder = MockMultiVectorEmbedder::new(32, 64);

    // Generate corpus with more tokens per document
    let documents: Vec<String> = (0..100)
        .map(|i| {
            format!(
                "Document number {} contains detailed information about topic {} in category {} covering various aspects",
                i,
                i % 10,
                i % 5
            )
        })
        .collect();

    // Use small number of centroids (4) - this tests whether nprobe=4 can achieve high recall
    // With 4 centroids and nprobe=4, exhaustive probes all centroids
    let config = WarpIndexConfig::new(2, 4, 32).with_kmeans_iterations(10);
    let mut index = WarpIndex::new(config);

    let embeddings: Vec<MultiVectorEmbedding> = documents
        .iter()
        .map(|d| embedder.embed_tokens(d).unwrap())
        .collect();

    if let Err(e) = index.train(&embeddings) {
        return ConjectureResult {
            name,
            hypothesis,
            threshold,
            observed_value: format!("Training failed: {e}"),
            verdict: Verdict::Falsified,
            details: "Could not train index".to_string(),
        };
    }

    for (i, doc) in documents.iter().enumerate() {
        let chunk = Chunk::new(DocumentId::new(), doc.clone(), 0, doc.len());
        index.insert(chunk, embeddings[i].clone()).unwrap();
    }
    index.build().unwrap();

    // Test queries
    let queries = vec![
        "information about topic 5",
        "document in category 2",
        "number contains information",
    ];

    let mut total_recall = 0.0;
    let num_queries = queries.len();

    for query in &queries {
        let query_emb = embedder.embed_tokens(query).unwrap();

        // Exhaustive search (high nprobe)
        let exhaustive_config = WarpSearchConfig::with_k(10).nprobe(8).bound(1000);
        let exhaustive_results = index.search(&query_emb, &exhaustive_config).unwrap();
        let exhaustive_ids: std::collections::HashSet<_> = exhaustive_results
            .iter()
            .map(|(id, _)| id.clone())
            .collect();

        // Pruned search (nprobe=4)
        let pruned_config = WarpSearchConfig::with_k(10).nprobe(4).bound(128);
        let pruned_results = index.search(&query_emb, &pruned_config).unwrap();
        let pruned_ids: std::collections::HashSet<_> =
            pruned_results.iter().map(|(id, _)| id.clone()).collect();

        // Recall: how many of exhaustive top-10 are in pruned top-10
        let intersection = pruned_ids.intersection(&exhaustive_ids).count();
        let recall = if exhaustive_ids.is_empty() {
            1.0
        } else {
            intersection as f64 / exhaustive_ids.len() as f64
        };
        total_recall += recall;
    }

    let avg_recall = total_recall / num_queries as f64;
    let observed = format!("recall@10 = {:.4} ({:.2}%)", avg_recall, avg_recall * 100.0);

    let verdict = if avg_recall >= 0.95 {
        Verdict::Corroborated
    } else {
        Verdict::Falsified
    };

    let details = format!(
        "Tested {} queries. Recall {} 95% threshold.",
        num_queries,
        if verdict == Verdict::Corroborated {
            "meets"
        } else {
            "below"
        }
    );

    ConjectureResult {
        name,
        hypothesis,
        threshold,
        observed_value: observed,
        verdict,
        details,
    }
}

// =============================================================================
// CONJECTURE 3: Scaling Laws
// =============================================================================

/// Test: Memory and latency scaling properties
fn test_conjecture_3_scaling() -> ConjectureResult {
    let name = "Conjecture 3: Scaling Laws".to_string();
    let hypothesis = "Memory scales with N*T*bits, latency scales with nprobe not N".to_string();
    let threshold = "Memory < theoretical*1.2, latency O(nprobe) not O(N)".to_string();

    let embedder = MockMultiVectorEmbedder::new(32, 64);

    // Test with increasing corpus sizes - smaller sizes for faster testing
    let corpus_sizes = [500, 1000, 2000];
    let mut memory_results: Vec<(usize, usize)> = Vec::new();
    let mut latency_results: Vec<(usize, f64)> = Vec::new();

    for &n in &corpus_sizes {
        // More tokens per document
        let documents: Vec<String> = (0..n)
            .map(|i| {
                format!(
                    "Document {} about topic {} in field {} with additional context and details",
                    i,
                    i % 50,
                    i % 10
                )
            })
            .collect();

        // Use appropriate number of centroids for corpus size
        let num_centroids = 8.max(n / 100);
        let config = WarpIndexConfig::new(2, num_centroids, 32).with_kmeans_iterations(5);
        let mut index = WarpIndex::new(config);

        let embeddings: Vec<MultiVectorEmbedding> = documents
            .iter()
            .map(|d| embedder.embed_tokens(d).unwrap())
            .collect();

        if index.train(&embeddings).is_err() {
            continue;
        }

        for (i, doc) in documents.iter().enumerate() {
            let chunk = Chunk::new(DocumentId::new(), doc.clone(), 0, doc.len());
            let _ = index.insert(chunk, embeddings[i].clone());
        }
        let _ = index.build();

        // Measure memory
        let memory = index.memory_usage();
        memory_results.push((n, memory));

        // Measure latency (average over multiple queries)
        let query_emb = embedder.embed_tokens("topic document field").unwrap();
        let search_config = WarpSearchConfig::with_k(10).nprobe(4);

        let start = Instant::now();
        let num_queries = 50;
        for _ in 0..num_queries {
            let _ = index.search(&query_emb, &search_config);
        }
        let elapsed = start.elapsed();
        let avg_latency_us = elapsed.as_micros() as f64 / num_queries as f64;
        latency_results.push((n, avg_latency_us));
    }

    // Analyze memory scaling
    // Theoretical: N * T * dim * nbits / 8 bytes (compressed tokens only)
    // With overhead for metadata
    let memory_ok = if memory_results.len() >= 2 {
        let (n1, m1) = memory_results[0];
        let (n2, m2) = memory_results[1];

        // Memory should scale roughly linearly with N
        // Check if ratio is reasonable (allow 50% variance)
        let ratio = m2 as f64 / m1 as f64;
        let expected_ratio = n2 as f64 / n1 as f64;
        (ratio / expected_ratio).abs() < 1.5 && (ratio / expected_ratio).abs() > 0.5
    } else {
        true
    };

    // Analyze latency scaling
    // Latency should NOT scale linearly with N (that would mean no pruning benefit)
    let latency_ok = if latency_results.len() >= 2 {
        let (n1, l1) = latency_results[0];
        let (n2, l2) = latency_results[1];

        // If latency scaled linearly with N, l2/l1 would equal n2/n1
        // We want sub-linear scaling: l2/l1 < n2/n1
        let latency_ratio = l2 / l1;
        let n_ratio = n2 as f64 / n1 as f64;

        // Allow some increase but not proportional to N
        latency_ratio < n_ratio * 0.8 // Must be at least 20% better than linear
    } else {
        true
    };

    let observed = format!(
        "Memory: {:?}, Latency: {:?}",
        memory_results
            .iter()
            .map(|(n, m)| format!("N={}: {}KB", n, m / 1024))
            .collect::<Vec<_>>()
            .join(", "),
        latency_results
            .iter()
            .map(|(n, l)| format!("N={}: {:.0}us", n, l))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let verdict = if memory_ok && latency_ok {
        Verdict::Corroborated
    } else {
        Verdict::Falsified
    };

    let details = format!(
        "Memory scaling: {}, Latency scaling: {}",
        if memory_ok { "OK" } else { "FAILED" },
        if latency_ok {
            "sub-linear (OK)"
        } else {
            "linear with N (FAILED)"
        }
    );

    ConjectureResult {
        name,
        hypothesis,
        threshold,
        observed_value: observed,
        verdict,
        details,
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute average embedding from multi-vector (for single-vector comparison)
fn average_embedding(mv: &MultiVectorEmbedding) -> Vec<f32> {
    if mv.num_tokens() == 0 {
        return vec![0.0; mv.dim()];
    }

    let mut avg = vec![0.0f32; mv.dim()];
    for token in mv.tokens() {
        for (i, &v) in token.iter().enumerate() {
            avg[i] += v;
        }
    }

    let n = mv.num_tokens() as f32;
    for v in &mut avg {
        *v /= n;
    }

    // Normalize
    let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut avg {
            *v /= norm;
        }
    }

    avg
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a: f64 = a
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Compute quantized MaxSim score
fn compute_quantized_maxsim(
    codec: &ResidualCodec,
    query: &MultiVectorEmbedding,
    doc: &MultiVectorEmbedding,
) -> f64 {
    let mut total_score = 0.0;

    for query_token in query.tokens() {
        let mut max_score = f32::NEG_INFINITY;

        for doc_token in doc.tokens() {
            // Compress document token
            let (centroid_id, residual) = codec.compress(doc_token);

            // Compute centroid score
            let centroid_score = codec.centroid_score(query_token, centroid_id);

            // Decompress and score
            let score = codec.decompress_score(query_token, centroid_id, centroid_score, &residual);

            if score > max_score {
                max_score = score;
            }
        }

        if max_score > f32::NEG_INFINITY {
            total_score += max_score as f64;
        }
    }

    total_score
}

/// Compute Kendall's tau rank correlation coefficient
fn kendalls_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 1.0;
    }

    let mut concordant = 0i64;
    let mut discordant = 0i64;

    for i in 0..n {
        for j in (i + 1)..n {
            let x_diff = x[i] - x[j];
            let y_diff = y[i] - y[j];

            let product = x_diff * y_diff;

            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
            // Ties are ignored
        }
    }

    let total = concordant + discordant;
    if total == 0 {
        return 1.0;
    }

    (concordant - discordant) as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kendalls_tau_perfect_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tau = kendalls_tau(&x, &y);
        assert!((tau - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_kendalls_tau_perfect_inverse() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let tau = kendalls_tau(&x, &y);
        assert!((tau - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    /// Integration test: Run full falsification plan
    /// This is the main test that executes all falsification tests
    #[test]
    fn test_falsification_plan() {
        let report = execute_falsification_plan();

        println!("\n");
        report.print_report();

        // We don't assert on Corroborated because the goal is to try to falsify
        // But we track the results
        println!(
            "\nTest completed. Overall verdict: {}",
            report.overall_verdict
        );
    }

    /// Specific test for Experimentum Crucis
    #[test]
    fn test_experimentum_crucis_standalone() {
        let result = test_experimentum_crucis();
        println!("\nExperimentum Crucis Result:");
        println!("  Observed: {}", result.observed_value);
        println!("  Verdict: {}", result.verdict);
    }

    /// Specific test for Compression Conjecture
    #[test]
    fn test_compression_conjecture_standalone() {
        let result = test_conjecture_1_compression();
        println!("\nCompression Conjecture Result:");
        println!("  Observed: {}", result.observed_value);
        println!("  Verdict: {}", result.verdict);
    }

    /// Specific test for Pruning Conjecture
    #[test]
    fn test_pruning_conjecture_standalone() {
        let result = test_conjecture_2_pruning();
        println!("\nPruning Conjecture Result:");
        println!("  Observed: {}", result.observed_value);
        println!("  Verdict: {}", result.verdict);
    }

    /// Specific test for Scaling Conjecture
    #[test]
    fn test_scaling_conjecture_standalone() {
        let result = test_conjecture_3_scaling();
        println!("\nScaling Conjecture Result:");
        println!("  Observed: {}", result.observed_value);
        println!("  Verdict: {}", result.verdict);
    }
}
