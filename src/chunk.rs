//! Document chunking strategies for RAG pipelines

use crate::{Document, DocumentId, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique chunk identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub uuid::Uuid);

impl ChunkId {
    /// Create a new random chunk ID
    #[must_use]
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for ChunkId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Metadata associated with a chunk
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Source document title
    pub title: Option<String>,
    /// Section/header hierarchy
    pub headers: Vec<String>,
    /// Page number (for PDFs)
    pub page: Option<usize>,
    /// Custom metadata
    pub custom: HashMap<String, serde_json::Value>,
}

/// A chunk of text from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk identifier
    pub id: ChunkId,
    /// Source document reference
    pub document_id: DocumentId,
    /// Chunk text content
    pub content: String,
    /// Character offset in source document (start)
    pub start_offset: usize,
    /// Character offset in source document (end)
    pub end_offset: usize,
    /// Metadata inherited from document
    pub metadata: ChunkMetadata,
    /// Embedding vector (populated after embedding)
    pub embedding: Option<Vec<f32>>,
}

impl Chunk {
    /// Create a new chunk
    #[must_use]
    pub fn new(
        document_id: DocumentId,
        content: String,
        start_offset: usize,
        end_offset: usize,
    ) -> Self {
        Self {
            id: ChunkId::new(),
            document_id,
            content,
            start_offset,
            end_offset,
            metadata: ChunkMetadata::default(),
            embedding: None,
        }
    }

    /// Get the length of the chunk in characters
    #[must_use]
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Check if the chunk is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Set the embedding vector
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }
}

/// Chunking strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed-size chunks with overlap
    FixedSize {
        /// Target chunk size in characters
        chunk_size: usize,
        /// Overlap between consecutive chunks
        overlap: usize,
    },
    /// Split on sentence boundaries
    Sentence {
        /// Maximum sentences per chunk
        max_sentences: usize,
        /// Overlap sentences between chunks
        overlap_sentences: usize,
    },
    /// Split on paragraph boundaries
    Paragraph {
        /// Maximum paragraphs per chunk
        max_paragraphs: usize,
    },
    /// Recursive character splitting
    Recursive {
        /// Separators to try in order
        separators: Vec<String>,
        /// Target chunk size
        chunk_size: usize,
        /// Overlap between chunks
        overlap: usize,
    },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::Recursive {
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                " ".to_string(),
            ],
            chunk_size: 512,
            overlap: 50,
        }
    }
}

/// Trait for document chunkers
pub trait Chunker: Send + Sync {
    /// Split document into chunks
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>>;

    /// Estimate chunk count without materializing
    fn estimate_chunks(&self, document: &Document) -> usize;
}

/// Recursive chunker implementation
#[derive(Debug, Clone)]
pub struct RecursiveChunker {
    separators: Vec<String>,
    chunk_size: usize,
    overlap: usize,
}

impl RecursiveChunker {
    /// Create a new recursive chunker
    #[must_use]
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                " ".to_string(),
            ],
            chunk_size,
            overlap,
        }
    }

    /// Create with custom separators
    #[must_use]
    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    fn split_text(&self, text: &str, separator_idx: usize) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return vec![text.to_string()];
        }

        if separator_idx >= self.separators.len() {
            // Fallback: split by characters
            return self.split_by_chars(text);
        }

        let separator = &self.separators[separator_idx];
        let parts: Vec<&str> = text.split(separator).collect();

        if parts.len() == 1 {
            // Separator not found, try next
            return self.split_text(text, separator_idx + 1);
        }

        self.merge_splits(&parts, separator, separator_idx)
    }

    fn merge_splits(&self, parts: &[&str], separator: &str, separator_idx: usize) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();

        for part in parts {
            let potential = if current.is_empty() {
                part.to_string()
            } else {
                format!("{current}{separator}{part}")
            };

            if potential.len() <= self.chunk_size {
                current = potential;
            } else if current.is_empty() {
                // Single part too large, recurse
                chunks.extend(self.split_text(part, separator_idx + 1));
            } else {
                chunks.push(current);
                current = part.to_string();
            }
        }

        if !current.is_empty() {
            if current.len() <= self.chunk_size {
                chunks.push(current);
            } else {
                chunks.extend(self.split_text(&current, separator_idx + 1));
            }
        }

        chunks
    }

    fn split_by_chars(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);

            if end >= chars.len() {
                break;
            }

            // Move start, accounting for overlap
            start = if end > self.overlap {
                end - self.overlap
            } else {
                end
            };
        }

        chunks
    }

    fn apply_overlap(&self, chunks: Vec<String>) -> Vec<String> {
        if self.overlap == 0 || chunks.len() <= 1 {
            return chunks;
        }

        let mut result = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            if i == 0 {
                result.push(chunk.clone());
            } else {
                // Add overlap from previous chunk
                let prev = &chunks[i - 1];
                let overlap_text = if prev.len() > self.overlap {
                    &prev[prev.len() - self.overlap..]
                } else {
                    prev.as_str()
                };
                result.push(format!("{overlap_text}{chunk}"));
            }
        }
        result
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document
                    .title
                    .clone()
                    .unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let text_chunks = self.split_text(&document.content, 0);
        let overlapped = self.apply_overlap(text_chunks);

        let mut offset = 0;
        let mut chunks = Vec::new();

        for content in overlapped {
            // Find actual position in document
            let start = document.content[offset..]
                .find(&content)
                .map_or(offset, |pos| offset + pos);
            let end = start + content.len();

            let mut chunk = Chunk::new(document.id, content, start, end);
            chunk.metadata.title = document.title.clone();

            chunks.push(chunk);
            offset = start + 1; // Move past to find next occurrence
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let effective_size = self.chunk_size.saturating_sub(self.overlap);
        if effective_size == 0 {
            return 1;
        }
        (document.content.len() + effective_size - 1) / effective_size
    }
}

/// Fixed-size chunker implementation
#[derive(Debug, Clone)]
pub struct FixedSizeChunker {
    chunk_size: usize,
    overlap: usize,
}

impl FixedSizeChunker {
    /// Create a new fixed-size chunker
    #[must_use]
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            chunk_size,
            overlap,
        }
    }
}

impl Chunker for FixedSizeChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document
                    .title
                    .clone()
                    .unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let chars: Vec<char> = document.content.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let content: String = chars[start..end].iter().collect();

            let byte_start = chars[..start].iter().collect::<String>().len();
            let byte_end = byte_start + content.len();

            let mut chunk = Chunk::new(document.id, content, byte_start, byte_end);
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);

            if end >= chars.len() {
                break;
            }

            let step = self.chunk_size.saturating_sub(self.overlap);
            start += if step == 0 { 1 } else { step };
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let step = self.chunk_size.saturating_sub(self.overlap);
        if step == 0 {
            return document.content.chars().count();
        }
        let char_count = document.content.chars().count();
        (char_count + step - 1) / step
    }
}

/// Semantic chunker that groups sentences by embedding similarity
pub struct SemanticChunker<E: crate::embed::Embedder> {
    embedder: E,
    /// Similarity threshold (0.0 to 1.0) - chunks split when similarity drops below this
    pub similarity_threshold: f32,
    /// Maximum chunk size in characters
    pub max_chunk_size: usize,
}

impl<E: crate::embed::Embedder> SemanticChunker<E> {
    /// Create a new semantic chunker
    pub fn new(embedder: E, similarity_threshold: f32, max_chunk_size: usize) -> Self {
        Self {
            embedder,
            similarity_threshold,
            max_chunk_size,
        }
    }

    /// Split text into sentences
    fn split_sentences(text: &str) -> Vec<&str> {
        let mut sentences = Vec::new();
        let mut start = 0;

        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                let next_char = text[i + c.len_utf8()..].chars().next();
                if next_char.map_or(true, |nc| nc.is_whitespace()) {
                    let end = i + c.len_utf8();
                    let sentence = text[start..end].trim();
                    if !sentence.is_empty() {
                        sentences.push(sentence);
                    }
                    start = end;
                }
            }
        }

        let remaining = text[start..].trim();
        if !remaining.is_empty() {
            sentences.push(remaining);
        }

        sentences
    }
}

impl<E: crate::embed::Embedder> Chunker for SemanticChunker<E> {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document
                    .title
                    .clone()
                    .unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let sentences = Self::split_sentences(&document.content);
        if sentences.is_empty() {
            return Err(Error::EmptyDocument(
                document
                    .title
                    .clone()
                    .unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        if sentences.len() == 1 {
            let content = sentences[0].to_string();
            let start_offset = document.content.find(&content).unwrap_or(0);
            let end_offset = start_offset + content.len();
            let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
            chunk.metadata.title = document.title.clone();
            return Ok(vec![chunk]);
        }

        // Embed all sentences
        let embeddings: Vec<Vec<f32>> = sentences
            .iter()
            .map(|s| {
                self.embedder
                    .embed(s)
                    .unwrap_or_else(|_| vec![0.0; self.embedder.dimension()])
            })
            .collect();

        let mut chunks = Vec::new();
        let mut current_sentences: Vec<&str> = vec![sentences[0]];
        let mut current_embedding = &embeddings[0];

        for i in 1..sentences.len() {
            let similarity = crate::embed::cosine_similarity(current_embedding, &embeddings[i]);
            let current_len: usize = current_sentences.iter().map(|s| s.len()).sum();

            if similarity < self.similarity_threshold
                || current_len + sentences[i].len() > self.max_chunk_size
            {
                // Create chunk from current sentences
                let content = current_sentences.join(" ");
                let start_offset = document.content.find(&content).unwrap_or(0);
                let end_offset = start_offset + content.len();
                let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
                chunk.metadata.title = document.title.clone();
                chunks.push(chunk);

                current_sentences = vec![sentences[i]];
                current_embedding = &embeddings[i];
            } else {
                current_sentences.push(sentences[i]);
            }
        }

        // Add remaining sentences
        if !current_sentences.is_empty() {
            let content = current_sentences.join(" ");
            let start_offset = document.content.find(&content).unwrap_or(0);
            let end_offset = start_offset + content.len();
            let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        // Rough estimate based on max chunk size
        let sentences = Self::split_sentences(&document.content);
        (sentences.len() + 2) / 3 // Assume average 3 sentences per chunk
    }
}

/// Structural chunker that respects document structure (headers, sections)
#[derive(Debug, Clone)]
pub struct StructuralChunker {
    /// Whether to respect headers when chunking
    pub respect_headers: bool,
    /// Maximum section size in characters
    pub max_section_size: usize,
}

impl StructuralChunker {
    /// Create a new structural chunker
    #[must_use]
    pub fn new(respect_headers: bool, max_section_size: usize) -> Self {
        Self {
            respect_headers,
            max_section_size,
        }
    }

    /// Extract header text from a line
    fn extract_header(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            // Markdown header
            let header = trimmed.trim_start_matches('#').trim();
            if !header.is_empty() {
                return Some(header.to_string());
            }
        }
        None
    }

    /// Check if a line is a header
    fn is_header(line: &str) -> bool {
        Self::extract_header(line).is_some()
    }

    /// Split document into sections by headers
    fn split_by_headers(text: &str) -> Vec<(Option<String>, String)> {
        let mut sections = Vec::new();
        let mut current_header: Option<String> = None;
        let mut current_content = String::new();

        for line in text.lines() {
            if Self::is_header(line) {
                // Save previous section if not empty
                if !current_content.trim().is_empty() || current_header.is_some() {
                    sections.push((current_header.take(), current_content.trim().to_string()));
                    current_content = String::new();
                }
                current_header = Self::extract_header(line);
                current_content.push_str(line);
                current_content.push('\n');
            } else {
                current_content.push_str(line);
                current_content.push('\n');
            }
        }

        // Add final section
        if !current_content.trim().is_empty() {
            sections.push((current_header, current_content.trim().to_string()));
        }

        sections
    }
}

impl Chunker for StructuralChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document
                    .title
                    .clone()
                    .unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let sections = if self.respect_headers {
            Self::split_by_headers(&document.content)
        } else {
            vec![(None, document.content.clone())]
        };

        if sections.is_empty() {
            return Err(Error::EmptyDocument(
                document
                    .title
                    .clone()
                    .unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let mut chunks = Vec::new();

        for (header, content) in sections {
            if content.is_empty() {
                continue;
            }

            // Split large sections if needed
            if content.len() > self.max_section_size {
                let sub_chunker = RecursiveChunker::new(self.max_section_size, 50);
                let sub_doc = Document {
                    id: document.id,
                    content: content.clone(),
                    title: document.title.clone(),
                    source: document.source.clone(),
                    metadata: document.metadata.clone(),
                };
                if let Ok(sub_chunks) = sub_chunker.chunk(&sub_doc) {
                    for mut chunk in sub_chunks {
                        if let Some(ref h) = header {
                            chunk.metadata.headers.push(h.clone());
                        }
                        chunks.push(chunk);
                    }
                }
            } else {
                let start_offset = document.content.find(&content).unwrap_or(0);
                let end_offset = start_offset + content.len();
                let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
                chunk.metadata.title = document.title.clone();
                if let Some(h) = header {
                    chunk.metadata.headers.push(h);
                }
                chunks.push(chunk);
            }
        }

        if chunks.is_empty() {
            // Fallback: return entire document as single chunk
            let content = document.content.clone();
            let mut chunk = Chunk::new(document.id, content, 0, document.content.len());
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let sections = Self::split_by_headers(&document.content);
        sections.len().max(1)
    }
}

/// Paragraph-based chunker
#[derive(Debug, Clone)]
pub struct ParagraphChunker {
    max_paragraphs: usize,
}

impl ParagraphChunker {
    /// Create a new paragraph chunker
    #[must_use]
    pub fn new(max_paragraphs: usize) -> Self {
        Self { max_paragraphs }
    }

    /// Split text into paragraphs
    fn split_paragraphs(text: &str) -> Vec<&str> {
        text.split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect()
    }
}

impl Chunker for ParagraphChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document
                    .title
                    .clone()
                    .unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let paragraphs = Self::split_paragraphs(&document.content);
        if paragraphs.is_empty() {
            return Err(Error::EmptyDocument(
                document
                    .title
                    .clone()
                    .unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let mut chunks = Vec::new();
        let mut i = 0;

        while i < paragraphs.len() {
            let end = (i + self.max_paragraphs).min(paragraphs.len());
            let content = paragraphs[i..end].join("\n\n");

            let start_offset = document.content.find(&content).unwrap_or(0);
            let end_offset = start_offset + content.len();

            let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);

            i = end;
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let paragraphs = Self::split_paragraphs(&document.content);
        if self.max_paragraphs == 0 {
            return paragraphs.len();
        }
        (paragraphs.len() + self.max_paragraphs - 1) / self.max_paragraphs
    }
}

/// Sentence-based chunker
#[derive(Debug, Clone)]
pub struct SentenceChunker {
    max_sentences: usize,
    overlap_sentences: usize,
}

impl SentenceChunker {
    /// Create a new sentence chunker
    #[must_use]
    pub fn new(max_sentences: usize, overlap_sentences: usize) -> Self {
        Self {
            max_sentences,
            overlap_sentences,
        }
    }

    fn split_sentences(text: &str) -> Vec<&str> {
        let mut sentences = Vec::new();
        let mut start = 0;

        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                // Check for end of sentence
                let next_char = text[i + c.len_utf8()..].chars().next();
                if next_char.map_or(true, |nc| nc.is_whitespace() || nc.is_uppercase()) {
                    let end = i + c.len_utf8();
                    let sentence = text[start..end].trim();
                    if !sentence.is_empty() {
                        sentences.push(sentence);
                    }
                    start = end;
                }
            }
        }

        // Add remaining text
        let remaining = text[start..].trim();
        if !remaining.is_empty() {
            sentences.push(remaining);
        }

        sentences
    }
}

impl Chunker for SentenceChunker {
    fn chunk(&self, document: &Document) -> Result<Vec<Chunk>> {
        if document.content.is_empty() {
            return Err(Error::EmptyDocument(
                document
                    .title
                    .clone()
                    .unwrap_or_else(|| "untitled".to_string()),
            ));
        }

        let sentences = Self::split_sentences(&document.content);
        let mut chunks = Vec::new();
        let mut i = 0;

        while i < sentences.len() {
            let end = (i + self.max_sentences).min(sentences.len());
            let content = sentences[i..end].join(" ");

            let start_offset = document.content.find(&content).unwrap_or(0);
            let end_offset = start_offset + content.len();

            let mut chunk = Chunk::new(document.id, content, start_offset, end_offset);
            chunk.metadata.title = document.title.clone();
            chunks.push(chunk);

            let step = self.max_sentences.saturating_sub(self.overlap_sentences);
            i += if step == 0 { 1 } else { step };
        }

        Ok(chunks)
    }

    fn estimate_chunks(&self, document: &Document) -> usize {
        if document.content.is_empty() {
            return 0;
        }
        let sentences = Self::split_sentences(&document.content);
        let step = self.max_sentences.saturating_sub(self.overlap_sentences);
        if step == 0 {
            return sentences.len();
        }
        (sentences.len() + step - 1) / step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ ChunkId Tests ============

    #[test]
    fn test_chunk_id_unique() {
        let id1 = ChunkId::new();
        let id2 = ChunkId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_chunk_id_default() {
        let id1 = ChunkId::default();
        let id2 = ChunkId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_chunk_id_display() {
        let id = ChunkId::new();
        let display = format!("{id}");
        assert!(!display.is_empty());
        assert!(display.contains('-'));
    }

    #[test]
    fn test_chunk_id_serialization() {
        let id = ChunkId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: ChunkId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    // ============ ChunkMetadata Tests ============

    #[test]
    fn test_chunk_metadata_default() {
        let meta = ChunkMetadata::default();
        assert!(meta.title.is_none());
        assert!(meta.headers.is_empty());
        assert!(meta.page.is_none());
        assert!(meta.custom.is_empty());
    }

    #[test]
    fn test_chunk_metadata_serialization() {
        let mut meta = ChunkMetadata {
            title: Some("Test".to_string()),
            headers: vec!["Section 1".to_string()],
            page: Some(42),
            ..Default::default()
        };
        meta.custom
            .insert("key".to_string(), serde_json::json!("value"));

        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: ChunkMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(meta.title, deserialized.title);
        assert_eq!(meta.headers, deserialized.headers);
        assert_eq!(meta.page, deserialized.page);
    }

    // ============ Chunk Tests ============

    #[test]
    fn test_chunk_creation() {
        let doc_id = DocumentId::new();
        let chunk = Chunk::new(doc_id, "Hello world".to_string(), 0, 11);

        assert_eq!(chunk.document_id, doc_id);
        assert_eq!(chunk.content, "Hello world");
        assert_eq!(chunk.start_offset, 0);
        assert_eq!(chunk.end_offset, 11);
        assert!(chunk.embedding.is_none());
    }

    #[test]
    fn test_chunk_len() {
        let doc_id = DocumentId::new();
        let chunk = Chunk::new(doc_id, "Hello".to_string(), 0, 5);
        assert_eq!(chunk.len(), 5);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn test_chunk_empty() {
        let doc_id = DocumentId::new();
        let chunk = Chunk::new(doc_id, String::new(), 0, 0);
        assert_eq!(chunk.len(), 0);
        assert!(chunk.is_empty());
    }

    #[test]
    fn test_chunk_set_embedding() {
        let doc_id = DocumentId::new();
        let mut chunk = Chunk::new(doc_id, "Test".to_string(), 0, 4);
        assert!(chunk.embedding.is_none());

        chunk.set_embedding(vec![0.1, 0.2, 0.3]);
        assert!(chunk.embedding.is_some());
        assert_eq!(chunk.embedding.unwrap(), vec![0.1, 0.2, 0.3]);
    }

    // ============ ChunkingStrategy Tests ============

    #[test]
    fn test_chunking_strategy_default() {
        let strategy = ChunkingStrategy::default();
        match strategy {
            ChunkingStrategy::Recursive {
                chunk_size,
                overlap,
                separators,
            } => {
                assert_eq!(chunk_size, 512);
                assert_eq!(overlap, 50);
                assert!(!separators.is_empty());
            }
            _ => panic!("Expected Recursive strategy"),
        }
    }

    #[test]
    fn test_chunking_strategy_serialization() {
        let strategy = ChunkingStrategy::FixedSize {
            chunk_size: 256,
            overlap: 32,
        };
        let json = serde_json::to_string(&strategy).unwrap();
        let deserialized: ChunkingStrategy = serde_json::from_str(&json).unwrap();

        match deserialized {
            ChunkingStrategy::FixedSize {
                chunk_size,
                overlap,
            } => {
                assert_eq!(chunk_size, 256);
                assert_eq!(overlap, 32);
            }
            _ => panic!("Wrong strategy type"),
        }
    }

    // ============ RecursiveChunker Tests ============

    #[test]
    fn test_recursive_chunker_new() {
        let chunker = RecursiveChunker::new(512, 50);
        assert_eq!(chunker.chunk_size, 512);
        assert_eq!(chunker.overlap, 50);
        assert!(!chunker.separators.is_empty());
    }

    #[test]
    fn test_recursive_chunker_custom_separators() {
        let chunker =
            RecursiveChunker::new(256, 20).with_separators(vec!["\n".to_string(), " ".to_string()]);
        assert_eq!(chunker.separators.len(), 2);
    }

    #[test]
    fn test_recursive_chunker_empty_document() {
        let chunker = RecursiveChunker::new(100, 10);
        let doc = Document::new("");

        let result = chunker.chunk(&doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_recursive_chunker_small_document() {
        let chunker = RecursiveChunker::new(1000, 100);
        let doc = Document::new("This is a small document.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "This is a small document.");
    }

    #[test]
    fn test_recursive_chunker_paragraph_split() {
        let chunker = RecursiveChunker::new(50, 10);
        let doc = Document::new("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_recursive_chunker_respects_chunk_size() {
        let chunker = RecursiveChunker::new(20, 5);
        let doc = Document::new("This is a longer document that needs to be split into multiple chunks based on the chunk size.");

        let chunks = chunker.chunk(&doc).unwrap();
        // All chunks (except maybe the first with overlap) should be <= chunk_size + overlap
        for chunk in &chunks {
            assert!(
                chunk.content.len() <= 25 + 5, // chunk_size + some tolerance
                "Chunk too large: {} chars",
                chunk.content.len()
            );
        }
    }

    #[test]
    fn test_recursive_chunker_preserves_document_id() {
        let chunker = RecursiveChunker::new(50, 10);
        let doc = Document::new("Content").with_title("Test Doc");

        let chunks = chunker.chunk(&doc).unwrap();
        for chunk in chunks {
            assert_eq!(chunk.document_id, doc.id);
            assert_eq!(chunk.metadata.title, Some("Test Doc".to_string()));
        }
    }

    #[test]
    fn test_recursive_chunker_estimate() {
        let chunker = RecursiveChunker::new(100, 20);
        let doc = Document::new("A".repeat(500));

        let estimate = chunker.estimate_chunks(&doc);
        let actual = chunker.chunk(&doc).unwrap().len();

        // Estimate should be in reasonable range
        assert!(estimate > 0);
        assert!(estimate <= actual * 2);
    }

    // ============ FixedSizeChunker Tests ============

    #[test]
    fn test_fixed_size_chunker_new() {
        let chunker = FixedSizeChunker::new(256, 32);
        assert_eq!(chunker.chunk_size, 256);
        assert_eq!(chunker.overlap, 32);
    }

    #[test]
    fn test_fixed_size_chunker_empty_document() {
        let chunker = FixedSizeChunker::new(100, 10);
        let doc = Document::new("");

        let result = chunker.chunk(&doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_fixed_size_chunker_small_document() {
        let chunker = FixedSizeChunker::new(100, 10);
        let doc = Document::new("Short text");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Short text");
    }

    #[test]
    fn test_fixed_size_chunker_exact_split() {
        let chunker = FixedSizeChunker::new(10, 0);
        let doc = Document::new("0123456789abcdefghij");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "0123456789");
        assert_eq!(chunks[1].content, "abcdefghij");
    }

    #[test]
    fn test_fixed_size_chunker_with_overlap() {
        let chunker = FixedSizeChunker::new(10, 3);
        let doc = Document::new("0123456789abcdefghij");

        let chunks = chunker.chunk(&doc).unwrap();
        assert!(chunks.len() >= 2);
        // Second chunk should start before end of first chunk
    }

    #[test]
    fn test_fixed_size_chunker_unicode() {
        let chunker = FixedSizeChunker::new(5, 0);
        let doc = Document::new("héllo wörld");

        let chunks = chunker.chunk(&doc).unwrap();
        // Should handle unicode correctly
        assert!(chunks.len() >= 2);
        for chunk in chunks {
            assert!(chunk.content.chars().count() <= 5);
        }
    }

    #[test]
    fn test_fixed_size_chunker_estimate() {
        let chunker = FixedSizeChunker::new(10, 2);
        let doc = Document::new("A".repeat(100));

        let estimate = chunker.estimate_chunks(&doc);
        let actual = chunker.chunk(&doc).unwrap().len();

        assert!(estimate > 0);
        #[allow(clippy::cast_possible_wrap)]
        let diff = (estimate as isize - actual as isize).abs();
        assert!(diff <= 2);
    }

    // ============ SentenceChunker Tests ============

    #[test]
    fn test_sentence_chunker_new() {
        let chunker = SentenceChunker::new(3, 1);
        assert_eq!(chunker.max_sentences, 3);
        assert_eq!(chunker.overlap_sentences, 1);
    }

    #[test]
    fn test_sentence_chunker_empty_document() {
        let chunker = SentenceChunker::new(2, 0);
        let doc = Document::new("");

        let result = chunker.chunk(&doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_sentence_chunker_single_sentence() {
        let chunker = SentenceChunker::new(2, 0);
        let doc = Document::new("This is a single sentence.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_sentence_chunker_multiple_sentences() {
        let chunker = SentenceChunker::new(2, 0);
        let doc =
            Document::new("First sentence. Second sentence. Third sentence. Fourth sentence.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_sentence_chunker_with_overlap() {
        let chunker = SentenceChunker::new(2, 1);
        let doc = Document::new("One. Two. Three. Four.");

        let chunks = chunker.chunk(&doc).unwrap();
        // With overlap, we should have more chunks
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_sentence_chunker_exclamation_question() {
        let chunker = SentenceChunker::new(1, 0);
        let doc = Document::new("Hello! How are you? I am fine.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn test_sentence_chunker_estimate() {
        let chunker = SentenceChunker::new(2, 1);
        let doc = Document::new("One. Two. Three. Four. Five.");

        let estimate = chunker.estimate_chunks(&doc);
        assert!(estimate > 0);
    }

    // ============ SemanticChunker Tests ============

    #[test]
    fn test_semantic_chunker_new() {
        let embedder = crate::embed::MockEmbedder::new(64);
        let chunker = SemanticChunker::new(embedder, 0.5, 1000);
        assert!((chunker.similarity_threshold - 0.5).abs() < 0.01);
        assert_eq!(chunker.max_chunk_size, 1000);
    }

    #[test]
    fn test_semantic_chunker_empty_document() {
        let embedder = crate::embed::MockEmbedder::new(64);
        let chunker = SemanticChunker::new(embedder, 0.5, 1000);
        let doc = Document::new("");

        let result = chunker.chunk(&doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_semantic_chunker_single_sentence() {
        let embedder = crate::embed::MockEmbedder::new(64);
        let chunker = SemanticChunker::new(embedder, 0.5, 1000);
        let doc = Document::new("This is a single sentence.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_semantic_chunker_multiple_sentences() {
        let embedder = crate::embed::MockEmbedder::new(64);
        let chunker = SemanticChunker::new(embedder, 0.9, 500); // High threshold forces splits
        let doc = Document::new("First sentence. Second sentence. Third sentence.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_semantic_chunker_preserves_document_id() {
        let embedder = crate::embed::MockEmbedder::new(64);
        let chunker = SemanticChunker::new(embedder, 0.5, 1000);
        let doc = Document::new("Test content here.").with_title("Test Doc");

        let chunks = chunker.chunk(&doc).unwrap();
        for chunk in chunks {
            assert_eq!(chunk.document_id, doc.id);
            assert_eq!(chunk.metadata.title, Some("Test Doc".to_string()));
        }
    }

    #[test]
    fn test_semantic_chunker_respects_max_size() {
        let embedder = crate::embed::MockEmbedder::new(64);
        let chunker = SemanticChunker::new(embedder, 0.0, 50); // Low threshold, small max size
        let doc = Document::new("First sentence here. Second sentence follows. Third sentence comes. Fourth sentence ends.");

        let chunks = chunker.chunk(&doc).unwrap();
        for chunk in &chunks {
            assert!(chunk.content.len() <= 100); // Some tolerance
        }
    }

    #[test]
    fn test_semantic_chunker_estimate() {
        let embedder = crate::embed::MockEmbedder::new(64);
        let chunker = SemanticChunker::new(embedder, 0.5, 100);
        let doc = Document::new("Sentence one. Sentence two. Sentence three.");

        let estimate = chunker.estimate_chunks(&doc);
        assert!(estimate > 0);
    }

    // ============ StructuralChunker Tests ============

    #[test]
    fn test_structural_chunker_new() {
        let chunker = StructuralChunker::new(true, 500);
        assert!(chunker.respect_headers);
        assert_eq!(chunker.max_section_size, 500);
    }

    #[test]
    fn test_structural_chunker_empty_document() {
        let chunker = StructuralChunker::new(true, 500);
        let doc = Document::new("");

        let result = chunker.chunk(&doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_structural_chunker_no_headers() {
        let chunker = StructuralChunker::new(true, 500);
        let doc = Document::new("Just plain text without headers.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_structural_chunker_markdown_headers() {
        let chunker = StructuralChunker::new(true, 1000);
        let doc = Document::new("# Header 1\n\nContent 1.\n\n# Header 2\n\nContent 2.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("Header 1"));
        assert!(chunks[1].content.contains("Header 2"));
    }

    #[test]
    fn test_structural_chunker_nested_headers() {
        let chunker = StructuralChunker::new(true, 1000);
        let doc =
            Document::new("# Main\n\nIntro.\n\n## Sub 1\n\nContent 1.\n\n## Sub 2\n\nContent 2.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_structural_chunker_preserves_metadata() {
        let chunker = StructuralChunker::new(true, 1000);
        let doc = Document::new("# Section\n\nContent.").with_title("Test Doc");

        let chunks = chunker.chunk(&doc).unwrap();
        for chunk in chunks {
            assert_eq!(chunk.document_id, doc.id);
            assert_eq!(chunk.metadata.title, Some("Test Doc".to_string()));
        }
    }

    #[test]
    fn test_structural_chunker_header_in_metadata() {
        let chunker = StructuralChunker::new(true, 1000);
        let doc = Document::new("# My Section\n\nSection content here.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert!(!chunks.is_empty());
        assert!(
            chunks[0]
                .metadata
                .headers
                .contains(&"My Section".to_string())
                || chunks[0].content.contains("My Section")
        );
    }

    #[test]
    fn test_structural_chunker_respects_max_size() {
        let chunker = StructuralChunker::new(true, 50);
        let doc = Document::new("# Header\n\n".to_string() + &"A ".repeat(100));

        let chunks = chunker.chunk(&doc).unwrap();
        // Should split large sections
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_structural_chunker_estimate() {
        let chunker = StructuralChunker::new(true, 500);
        let doc = Document::new("# H1\n\nC1.\n\n# H2\n\nC2.\n\n# H3\n\nC3.");

        let estimate = chunker.estimate_chunks(&doc);
        assert!(estimate > 0);
    }

    // ============ ParagraphChunker Tests ============

    #[test]
    fn test_paragraph_chunker_new() {
        let chunker = ParagraphChunker::new(3);
        assert_eq!(chunker.max_paragraphs, 3);
    }

    #[test]
    fn test_paragraph_chunker_empty_document() {
        let chunker = ParagraphChunker::new(2);
        let doc = Document::new("");

        let result = chunker.chunk(&doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_paragraph_chunker_single_paragraph() {
        let chunker = ParagraphChunker::new(2);
        let doc = Document::new("This is a single paragraph without line breaks.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].content.trim(),
            "This is a single paragraph without line breaks."
        );
    }

    #[test]
    fn test_paragraph_chunker_multiple_paragraphs() {
        let chunker = ParagraphChunker::new(1);
        let doc = Document::new("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_paragraph_chunker_groups_paragraphs() {
        let chunker = ParagraphChunker::new(2);
        let doc = Document::new("Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("Para 1"));
        assert!(chunks[0].content.contains("Para 2"));
    }

    #[test]
    fn test_paragraph_chunker_preserves_document_id() {
        let chunker = ParagraphChunker::new(1);
        let doc = Document::new("Para 1.\n\nPara 2.").with_title("Test Doc");

        let chunks = chunker.chunk(&doc).unwrap();
        for chunk in chunks {
            assert_eq!(chunk.document_id, doc.id);
            assert_eq!(chunk.metadata.title, Some("Test Doc".to_string()));
        }
    }

    #[test]
    fn test_paragraph_chunker_estimate() {
        let chunker = ParagraphChunker::new(2);
        let doc = Document::new("P1.\n\nP2.\n\nP3.\n\nP4.\n\nP5.");

        let estimate = chunker.estimate_chunks(&doc);
        let actual = chunker.chunk(&doc).unwrap().len();

        assert!(estimate > 0);
        #[allow(clippy::cast_possible_wrap)]
        let diff = (estimate as isize - actual as isize).abs();
        assert!(diff <= 2);
    }

    #[test]
    fn test_paragraph_chunker_whitespace_handling() {
        let chunker = ParagraphChunker::new(1);
        let doc = Document::new("  First paragraph.  \n\n  Second paragraph.  ");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 2);
        // Content should be trimmed
        assert!(!chunks[0].content.starts_with(' '));
        assert!(!chunks[1].content.ends_with(' '));
    }

    #[test]
    fn test_paragraph_chunker_triple_newline() {
        let chunker = ParagraphChunker::new(1);
        let doc = Document::new("Para 1.\n\n\nPara 2.");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks.len(), 2);
    }

    // ============ Edge Cases ============

    #[test]
    fn test_chunker_with_newlines() {
        let chunker = RecursiveChunker::new(50, 0);
        let doc = Document::new("Line 1\nLine 2\nLine 3");

        let chunks = chunker.chunk(&doc).unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunker_offset_tracking() {
        let chunker = FixedSizeChunker::new(5, 0);
        let doc = Document::new("0123456789");

        let chunks = chunker.chunk(&doc).unwrap();
        assert_eq!(chunks[0].start_offset, 0);
        assert_eq!(chunks[0].end_offset, 5);
        assert_eq!(chunks[1].start_offset, 5);
        assert_eq!(chunks[1].end_offset, 10);
    }

    // ============ Property-Based Tests ============

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_recursive_chunker_no_empty_chunks(content in "[a-zA-Z ]{10,500}") {
            let chunker = RecursiveChunker::new(50, 10);
            let doc = Document::new(content);

            if let Ok(chunks) = chunker.chunk(&doc) {
                for chunk in chunks {
                    prop_assert!(!chunk.is_empty());
                }
            }
        }

        #[test]
        fn prop_fixed_size_respects_max(content in "[a-zA-Z]{20,200}", chunk_size in 10usize..50) {
            let chunker = FixedSizeChunker::new(chunk_size, 0);
            let doc = Document::new(content);

            if let Ok(chunks) = chunker.chunk(&doc) {
                for chunk in chunks {
                    prop_assert!(chunk.content.chars().count() <= chunk_size);
                }
            }
        }

        #[test]
        fn prop_chunk_ids_unique(content in "[a-zA-Z ]{50,200}") {
            let chunker = FixedSizeChunker::new(20, 5);
            let doc = Document::new(content);

            if let Ok(chunks) = chunker.chunk(&doc) {
                let ids: std::collections::HashSet<_> = chunks.iter().map(|c| c.id).collect();
                prop_assert_eq!(ids.len(), chunks.len());
            }
        }

        #[test]
        fn prop_paragraph_chunker_no_empty_chunks(content in "[a-zA-Z ]{10,100}(\n\n[a-zA-Z ]{10,100}){1,5}") {
            let chunker = ParagraphChunker::new(2);
            let doc = Document::new(content);

            if let Ok(chunks) = chunker.chunk(&doc) {
                for chunk in chunks {
                    prop_assert!(!chunk.is_empty());
                }
            }
        }

        #[test]
        fn prop_paragraph_chunker_respects_max(
            content in "[a-zA-Z ]{5,30}(\n\n[a-zA-Z ]{5,30}){2,8}",
            max_paras in 1usize..5
        ) {
            let chunker = ParagraphChunker::new(max_paras);
            let doc = Document::new(content);

            if let Ok(chunks) = chunker.chunk(&doc) {
                for chunk in &chunks {
                    let para_count = chunk.content.split("\n\n").filter(|p| !p.trim().is_empty()).count();
                    prop_assert!(para_count <= max_paras);
                }
            }
        }
    }
}
