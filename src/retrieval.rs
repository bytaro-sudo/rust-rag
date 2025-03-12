//! Module for retrieving relevant chunks based on queries.

use crate::chunking::Chunk;
use crate::embedding::Embedding;
use crate::embedding::EmbeddingGenerator;
use anyhow::{anyhow, Result, Context};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Result of a retrieval operation.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// The retrieved chunk
    pub chunk: Chunk,
    /// The relevance score (typically cosine similarity)
    pub score: f32,
}

impl Eq for RetrievalResult {}

impl PartialEq for RetrievalResult {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

// For BinaryHeap we need to implement Ord
// We use reverse ordering to create a max-heap
impl Ord for RetrievalResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal).reverse()
    }
}

impl PartialOrd for RetrievalResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Interface for retrievers.
pub trait Retriever {
    /// Retrieve relevant chunks for a given query.
    fn retrieve(&self, query: &str, top_k:usize) -> Result<Vec<RetrievalResult>>;
}

/// A retriever that uses embedding similarity to find related chunks.
pub struct EmbeddingRetriever<'a> {
    chunks: Vec<Chunk>,
    embeddings: Vec<Embedding>,
    embedding_generator: &'a dyn EmbeddingGenerator,
}

impl<'a> EmbeddingRetriever<'a> {
    /// Create a new embedding-based retriever.
    pub fn new(
        chunks: Vec<Chunk>,
        embeddings: Vec<Embedding>,
        embedding_generator: &'a dyn EmbeddingGenerator,
    ) -> Result<Self> {
        // Validate that each chunk as a corresponding embedding
        if chunks.len() != embeddings.len() {
            return Err(anyhow!("Number of chunks ({}) does not match embeddings ({})",
            chunks.len(), embeddings.len()));
        }
        
        Ok(Self {
            chunks,
            embeddings,
            embedding_generator,
        })
    }
}

impl<'a> Retriever for EmbeddingRetriever<'a> {
    fn retrieve(&self, query: &str, top_k:usize) -> Result<Vec<RetrievalResult>> {
        // Generate embedding for the query
        let query_embedding = self.embedding_generator.generate(query)
            .context("Failed to generate embedding for query")?;
        
        // Use a binary heap to keep track if top_k results
        let mut heap = BinaryHeap::with_capacity(top_k + 1);
        
        // Calculate similarity for each chunk's embedding
        for (i, embedding) in self.embeddings.iter().enumerate() {
            let similarity = query_embedding.cosine_similarity(embedding);
            
            let result = RetrievalResult {
                chunk: self.chunks[i].clone(),
                score: similarity,
            };
            
            heap.push(result);
            
            // Keep only top_k results
            if heap.len() > top_k {
                heap.pop();
            }
        }
        
        // Convert heap to vector and reverse to get descending order
        let mut results: Vec<RetrievalResult> = heap.into_iter().collect();
        results.reverse();
        
        Ok(results)
    }
}

/// A simple retriever that searches for keywords in chunks.
pub struct KeywordRetriever {
    chunks: Vec<Chunk>,
}

impl KeywordRetriever {
    /// Create a new keyword-based retriever.
    pub fn new(chunks: Vec<Chunk>) -> Self {
        Self { chunks }
    }
    
    // Helper function to calculate a simple relevance score
    fn calculate_relevance(&self, query: &str, chunk_content: &str) -> f32 {
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let chunk_words: Vec<&str> = chunk_content.split_whitespace().collect();
        
        let mut score = 0.0;
        
        for query_word in &query_words {
            for chunk_word in &chunk_words {
                if chunk_word.to_lowercase().contains(&query_word.to_lowercase()) {
                    score += 1.0;
                }
            }    
        }
        
        // Normalize by the length of the query
        if !query_words.is_empty() {
            score /= query_words.len() as f32;
        }
        
        score
    }
}

impl Retriever for KeywordRetriever {
    fn retrieve(&self, query: &str, top_k:usize) -> Result<Vec<RetrievalResult>> {
        let mut results = Vec::new();
        
        // Calculate relevance score for each chunk
        for chunk in &self.chunks {
            let score = self.calculate_relevance(query, &chunk.content);
            
            // Only include chunks with some relevance
            if score > 0.0 {
                results.push(RetrievalResult {
                    chunk: chunk.clone(),
                    score
                })
            }
        }
        
        // Sort by score in descending order
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        
        // Return only top_k results
        Ok(results.into_iter().take(top_k).collect())
    }
}

/// Factory function to create a retriever based on available components.
pub fn create_retriever<'a>(
    chunks: Vec<Chunk>,
    embeddings: Option<Vec<Embedding>>,
    embedding_generator: Option<&'a dyn EmbeddingGenerator>,
) -> Result<Box<dyn Retriever + 'a>> {
    match (embeddings, embedding_generator) {
        (Some(embs), Some(generator)) => {
            let retriever = EmbeddingRetriever::new(chunks, embs, generator)?;
            Ok(Box::new(retriever))
        },
        _ => {
            // Fall back to keyword-based retrieval if embeddings are not available
            let retriever = KeywordRetriever::new(chunks);
            Ok(Box::new(retriever))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::Document;
    use crate::embedding::{MockEmbeddingGenerator, EmbeddingConfig, EmbeddingProvider, create_embedding_generator};
    use crate::chunking::{ChunkingStrategy, chunk_document};
    
    #[test]
    fn test_keyword_retriever() {
        // Create test chunks
        let chunks = vec![
            Chunk {
                id: "1".to_string(),
                content: "Rust is a systems programming language".to_string(),
                document_id: "doc1".to_string(),
                position: 0,
                metadata: Default::default(),
            },
            Chunk {
                id: "2".to_string(),
                content: "Python is a high-level programming language".to_string(),
                document_id: "doc1".to_string(),
                position: 1,
                metadata: Default::default(),
            }
        ];
        
        let retriever = KeywordRetriever::new(chunks);
        
        // Test retrieval
        let results = retriever.retrieve("rust systems", 2).unwrap();
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "1");
        assert!(results[0].score > 0.0);
        
        // Test retrieval with a more generic query
        let results = retriever.retrieve("programming language", 2).unwrap();
        
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_embedding_retriever() {
        // Create a document
        let doc = Document::new(
            "test-doc",
            "Rust is a systems programming language. Python is a high-level programming language."
        );

        // Chunk the document
        let chunks = chunk_document(&doc, &ChunkingStrategy::Sentence).unwrap();

        // Create embedding generator
        let config = EmbeddingConfig {
            provider: EmbeddingProvider::Mock { dimension: 64 },
        };
        let generator = create_embedding_generator(&config).unwrap();

        // Generate embeddings
        let embeddings = generator.embed_chunks(&chunks).unwrap();

        // Create retriever
        let retriever = EmbeddingRetriever::new(chunks, embeddings, generator.as_ref()).unwrap();

        // Test retrieval
        let results = retriever.retrieve("rust systems", 2).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].score >= 0.0 && results[0].score <= 1.0);
    }
}