//! Module for generating embeddings from text chunks.

use crate::chunking::Chunk;
use anyhow::{Result, Context, anyhow};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a vector embedding for a text chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Embedding {
    /// The vector representation
    pub vector: Vec<f32>,
    /// The ID of the chunk this embedding represents
    pub chunk_id: String,
    /// The dimension of the embedding vector
    pub dimension: usize,
    /// The model used to generate this embedding
    pub model: String,
}

impl Embedding {
    /// Creates a new embedding from a vector and associated metadata.
    pub fn new(vector: Vec<f32>, chunk_id: String, model: String) -> Self {
        let dimension = vector.len();
        Self {
            vector,
            chunk_id,
            dimension,
            model,
        }
    }

    /// Calculates the cosine similarity between this embedding and another.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.dimension != other.dimension {
            return 0.0; // Different dimensions can't be compared properly
        }

        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..self.dimension {
            dot_product += self.vector[i] * other.vector[i];
            norm_a += self.vector[i] * self.vector[i];
            norm_b += other.vector[i] * other.vector[i];
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a.sqrt() + norm_b.sqrt())
    }
}

/// Interface for embedding generators.
pub trait EmbeddingGenerator {
    /// Generate an embedding for a given text.
    fn generate(&self, text: &str) -> Result<Embedding>;

    /// Generate embeddings for multiple text in batch.
    fn generate_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        // Default implementation processes each test individually
        texts.iter()
            .map(|text| self.generate(text))
            .collect()
    }

    /// Generate embeddings for chunks.
    fn embed_chunks(&self, chunks: &[Chunk]) -> Result<Vec<Embedding>> {
        chunks.iter()
            .map(|chunk| {
                self.generate(&chunk.content)
                    .map(|mut embedding| {
                        embedding.chunk_id = chunk.id.clone();
                        embedding
                    })
            })
            .collect()
    }

    /// Returns the name of the model used by this generator.
    fn model_name(&self) -> &str;

    /// Returns the dimension of the embeddings generated.
    fn dimension(&self) -> usize;
}

/// A mock embedding generator for testing purposes.
#[derive(Debug)]
pub struct MockEmbeddingGenerator {
    pub dimension: usize,
    pub model_name: String,
}

impl MockEmbeddingGenerator {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            model_name: format!("mock_model_{dimension}d"),
        }
    }
}

impl EmbeddingGenerator for MockEmbeddingGenerator {
    fn generate(&self, text: &str) -> Result<Embedding> {
        // Generate a simple hash-based embedding for testing
        let vector = create_mock_embedding(text, self.dimension);

        Ok(Embedding::new(
            vector,
            format!("chunk_{}", text.len()), // Mock chunk ID
            self.model_name().to_string(),
        ))
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

// A simple function to create mock embeddings based on text hash
fn create_mock_embedding(text: &str, dimension: usize) -> Vec<f32> {
    let mut vector = Vec::with_capacity(dimension);
    let bytes = text.as_bytes();

    for i in 0..dimension {
        // Create a simple deterministic value based on character codes
        // This is only for testing and doesn't have any semantic meaning
        let sum: u32 = bytes.iter()
            .enumerate()
            .map(|(pos , &byte)| (byte as u32 * ((pos + i) % 255) as u32))
            .sum();

        let normalized = (sum % 1000) as f32 / 1000.0;
        vector.push(normalized);
    }

    // Normalize the vector to unit length
    let magnitude: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in &mut vector {
            *val /= magnitude;
        }
    }

    vector
}

/// Factory function to create an embedding generator based on configuration.
pub fn create_embedding_generator(config: &EmbeddingConfig) -> Result<Box<dyn EmbeddingGenerator>> {
    match &config.provider {
        EmbeddingProvider::Mock { dimension } => {
            Ok(Box::new(MockEmbeddingGenerator::new(*dimension)))
        },
        // We'll implement these in future iterations
        _ => Err(anyhow!("Embedding provider not implemented yet: {:?}", config.provider)),
    }
}

/// Configuration for embedding generation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingConfig {
    pub provider: EmbeddingProvider,
}

/// Available embdding providers.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EmbeddingProvider {
    /// API-based providers
    OpenAI {
        api_key: String,
        model_name: String,
    },
    HuggingFace {
        api_key: String,
        model_name: String,
    },
    /// Local model providers
    Local {
        model_path: String,
    },
    /// Mock provider for testing
    Mock {
        dimension: usize,
    }
}

impl fmt::Display for EmbeddingProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingProvider::OpenAI { model_name, ..} => write!(f, "OpenAI ({})", model_name),
            EmbeddingProvider::HuggingFace { model_name, .. } => write!(f, "HuggingFace ({})", model_name),
            EmbeddingProvider::Local { model_path, .. } => write!(f, "Local ({})", model_path),
            EmbeddingProvider::Mock { dimension, .. } => write!(f, "Mock ({})", dimension),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedding_generator() {
        let generator = MockEmbeddingGenerator::new(128);
        let embedding = generator.generate("This is a test document").unwrap();

        assert_eq!(embedding.dimension, 128);
        assert_eq!(embedding.vector.len(), 128);
    }

    #[test]
    fn test_cosine_similarity() {
        // Create two identical embeddings
        let text = "this is a test";
        let generator = MockEmbeddingGenerator::new(64);
        let embedding1 = generator.generate(text).unwrap();
        let embedding2 = generator.generate(text).unwrap();

        // Similarity to itself should be 1.0
        let similarity = embedding1.cosine_similarity(&embedding2);
        assert!((similarity - 1.0).abs() < 1e-6);

        // Create a different embedding
        let embedding3 = generator.generate("Completely different text").unwrap();
        let similarity = embedding1.cosine_similarity(&embedding3);

        // Different texts should have similarity < 1.0
        assert!(similarity < 1.0);
    }
}