//! Module for splitting documents into manageable chunks.

use std::collections::HashMap;
use anyhow::Result;

/// Represents a document to be processed in the RAG system
pub struct Document {
    /// Unique document identifier
    pub id: String,
    /// Textual content of the document
    pub content: String,
    /// Additional document metadata
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Creates a new document with basic metadata.
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Document {
            id: id.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Adds metadata to the document.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Represents a chunk of a document
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Unique chunk identifier
    pub id: String,
    /// Textual content of the chunk
    pub content: String,
    /// Reference to the source document
    pub document_id: String,
    /// Position of the chunk in the original document
    pub position: usize,
    /// Metadata inherited from the document and specific to the chunk
    pub metadata: HashMap<String, String>,
}

/// Strategy for splitting documents into chunks
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    /// Split by a fixed number of characters
    FixedSize(usize),
    /// Split by paragraphs
    Paragraph,
    /// Split by sentences
    Sentence,
}

/// Splits a document into chunks according to the specified strategy.
pub fn chunk_document(doc: &Document, strategy: &ChunkingStrategy) -> Result<Vec<Chunk>> {
    match strategy {
        ChunkingStrategy::FixedSize(size) => chunk_by_fixed_size(doc, *size),
        ChunkingStrategy::Paragraph => chunk_by_paragraph(doc),
        ChunkingStrategy::Sentence => chunk_by_sentence(doc)
    }
}

/// Basic implementation of fixed-size chunking
fn chunk_by_fixed_size(doc: &Document, size: usize) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    let mut position = 0;

    // Split content into chunks of approximate size
    let mut current_pos = 0;
    while current_pos < doc.content.len() {
        let end_pos = if current_pos + size >= doc.content.len() {
            doc.content.len()
        } else {
            // Look for the next space to avoid cutting words
            let slice = &doc.content[current_pos..(current_pos+size)];
            current_pos + slice.rfind(' ').unwrap_or(slice.len())
        };

        let chunk_content = doc.content[current_pos..end_pos].trim().to_string();

        // Only create a chunk if there's content
        if !chunk_content.is_empty() {
            let chunk = Chunk {
                id: format!("{}_{}", doc.id, position),
                content: chunk_content,
                document_id: doc.id.clone(),
                position,
                metadata: doc.metadata.clone(),
            };

            chunks.push(chunk);
            position += 1;
        }

        current_pos = end_pos;
    }

    Ok(chunks)
}

/// Basic implementation of paragraph-based chunking
fn chunk_by_paragraph(doc: &Document) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    let mut position = 0;

    // Split by paragraphs (empty lines)
    for paragraph in doc.content.split("\n\n") {
        let paragraph = paragraph.trim();
        if !paragraph.is_empty() {
            let chunk = Chunk {
                id: format!("{}_{}", doc.id, position),
                content: paragraph.to_string(),
                document_id: doc.id.clone(),
                position,
                metadata: doc.metadata.clone(),
            };

            chunks.push(chunk);
            position += 1;
        }
    }

    Ok(chunks)
}

/// Basic implementation of sentence-based chunking
/// This is a very simplified implementation.
fn chunk_by_sentence(doc: &Document) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    let mut position = 0;

    //Split by periods followed by space (simplification)
    for sentence in doc.content.split(". ") {
        let sentence = sentence.trim();
        if !sentence.is_empty() {
            let chunk = Chunk {
                id: format!("{}_{}", doc.id, position),
                content: format!("{}.", sentence),
                document_id: doc.id.clone(),
                position,
                metadata: doc.metadata.clone(),
            };

            chunks.push(chunk);
            position += 1;
        }
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_fixed_size_chunking() {
        let doc = Document::new(
            "test-doc",
            "This is a test document. It has several sentences. We want to see how chunking works."
        );

        let chunks = chunk_document(&doc, &ChunkingStrategy::FixedSize(20)).unwrap();

        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].document_id, "test-doc");
    }
}