//! Module for storing and searching vector embeddings.

use crate::chunking::Chunk;
use crate::embedding::Embedding;
use crate::embedding::EmbeddingGenerator;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use serde::{Serialize, Deserialize};

/// Storage interface for vector stores.
pub trait VectorStore {
    /// Add a single embedding with its associated chunk to the store.
    fn add(&mut self, embedding: Embedding, chunk: Chunk) -> Result<()>;

    /// Add multiple embeddings with their associated chunks to the store.
    fn add_many(&mut self, embeddings: Vec<Embedding>, chunks: Vec<Chunk>) -> Result<()> {
        if embeddings.len() != chunks.len() {
            return Err(anyhow!("Number of embeddings doesn't match number of chunks"));
        }

        for (embedding, chunk) in embeddings.into_iter().zip(chunks.into_iter()) {
            self.add(embedding, chunk)?;
        }

        Ok(())
    }

    /// Find the closest embeddings to the given query embedding.
    fn search(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<(Chunk, f32)>>;

    /// Get the total number of embeddings in the store.
    fn size(&self) -> usize;

    /// Clear all embeddings from the store.
    fn clear(&mut self) -> Result<()>;

    /// Save the store to disk.
    fn save(&self, path: &Path) -> Result<()>;

    /// Load a store from disk.
    fn load(&mut self, path: &Path) -> Result<()>;
}

/// Metadata for the in-memory vector store.
#[derive(Serialize, Deserialize)]
struct InMemoryStoreMetadata {
    version: String,
    embedding_model: String,
    embedding_dimension: usize,
    entry_count: usize,
}

/// A simple in-memory vector store.
#[derive(Default)]
pub struct InMemoryVectorStore {
    /// Map from chunk ID to chunk
    chunks: HashMap<String, Chunk>,
    /// Map from chunk ID to embedding
    embeddings: HashMap<String, Embedding>,
    /// Metadata about the store
    metadata: Option<InMemoryStoreMetadata>,
}

impl InMemoryVectorStore {
    /// Create a new in-memory vector store.
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            embeddings: HashMap::new(),
            metadata: None,
        }
    }

    /// Set metadata for the store.
    pub fn with_metadata(mut self, model: &str, dimension: usize) -> Self {
        self.metadata = Some(InMemoryStoreMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            embedding_model: model.to_string(),
            embedding_dimension: dimension,
            entry_count: 0,
        });

        self
    }
}

impl VectorStore for InMemoryVectorStore {
    fn add(&mut self, embedding: Embedding, chunk: Chunk) -> Result<()> {
        // Update metadata if exists
        if let Some(ref mut meta) = self.metadata {
            meta.entry_count += 1;

            // Validate embedding dimension
            if embedding.dimension != meta.embedding_dimension {
                return Err(anyhow!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    meta.embedding_dimension,
                    embedding.dimension
                ));
            }

            // Validate embedding model
            if embedding.model != meta.embedding_model {
                return Err(anyhow!(
                    "Embedding model mismatch: expected {}, got {}",
                    meta.embedding_model,
                    embedding.model
                ));
            }
        }

        // Store the chunk and embedding
        let chunk_id = chunk.id.clone();
        self.chunks.insert(chunk_id.clone(), chunk);
        self.embeddings.insert(chunk_id, embedding);

        Ok(())
    }

    fn search(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<(Chunk, f32)>> {
        // Calculate similarity for all embeddings
        let mut similarities: Vec<(String, f32)> = self.embeddings
            .iter()
            .map(|(id, embedding)| (id.clone(), query_embedding.cosine_similarity(embedding)))
            .collect();

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k results
        let top_results = similarities
            .into_iter()
            .take(top_k)
            .filter_map(|(id, score)| {
                self.chunks.get(&id).map(|chunk| (chunk.clone(), score))
            })
            .collect();

        Ok(top_results)
    }

    fn size(&self) -> usize {
        self.chunks.len()
    }

    fn clear(&mut self) -> Result<()> {
        self.chunks.clear();
        self.embeddings.clear();

        if let Some(ref mut meta) = self.metadata {
            meta.entry_count = 0;
        }

        Ok(())
    }

    fn save(&self, path: &Path) -> Result<()> {
        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Combine chunks and embeddings for serialization
        let entries: Vec<(Chunk, Embedding)> = self.chunks
            .iter()
            .filter_map(|(id, chunk)| {
                self.embeddings.get(id).map(|embedding| (chunk.clone(), embedding.clone()))
            })
            .collect();

        // Serialize metadata if available
        if let Some(ref meta) = self.metadata {
            let meta_path = path.with_file_name(format!(
                "{}.meta.json",
                path.file_name().unwrap().to_string_lossy()
            ));
            let meta_json = serde_json::to_string_pretty(meta)?;
            fs::write(&meta_path, meta_json)?;
        }

        // Serialize and save data
        let data = serde_json::to_string(&entries)?;
        fs::write(path, data)?;

        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        // Try to load metadata if available
        let meta_path = path.with_file_name(format!(
            "{}.meta.json",
            path.file_name().unwrap().to_string_lossy()
        ));

        if meta_path.exists() {
            let meta_json = fs::read_to_string(&meta_path)?;
            self.metadata = Some(serde_json::from_str(&meta_json)?);
        }

        // Load the main data
        let data = fs::read_to_string(path)?;
        let entries: Vec<(Chunk, Embedding)> = serde_json::from_str(&data)?;

        // Clear existing data
        self.clear()?;

        // Add all entries
        for (chunk, embedding) in entries {
            self.chunks.insert(chunk.id.clone(), chunk);
            self.embeddings.insert(embedding.chunk_id.clone(), embedding);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::MockEmbeddingGenerator;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn create_test_data() -> (Vec<Chunk>, Vec<Embedding>) {
        let chunks = vec![
            Chunk {
                id: "chunk1".to_string(),
                content: "This is the first test chunk".to_string(),
                document_id: "doc1".to_string(),
                position: 0,
                metadata: HashMap::new(),
            },
            Chunk {
                id: "chunk2".to_string(),
                content: "This is the second test chunk".to_string(),
                document_id: "doc1".to_string(),
                position: 1,
                metadata: HashMap::new(),
            },
        ];

        let generator = MockEmbeddingGenerator::new(64);

        let embeddings = chunks
            .iter()
            .map(|chunk| {
                let mut embedding = generator.generate(&chunk.content).unwrap();
                embedding.chunk_id = chunk.id.clone();
                embedding
            })
            .collect();

        (chunks, embeddings)
    }

    #[test]
    fn test_in_memory_vector_store() {
        let (chunks, embeddings) = create_test_data();

        let mut store = InMemoryVectorStore::new().with_metadata("mock_model_64d", 64);

        // Test add many
        store.add_many(embeddings.clone(), chunks.clone()).unwrap();
        assert_eq!(store.size(), 2);

        // Test search
        let query_embedding = embeddings[0].clone();
        let results = store.search(&query_embedding, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].1 >= results[1].1); // First result should have highest similarity

        // Test clear
        store.clear().unwrap();
        assert_eq!(store.size(), 0);
    }

    #[test]
    fn test_save_and_load() {
        let (chunks, embeddings) = create_test_data();

        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_store.json");

        // Create and populate store
        let mut store = InMemoryVectorStore::new().with_metadata("mock_model_64d", 64);
        store.add_many(embeddings, chunks).unwrap();

        // Save to file
        store.save(&file_path).unwrap();

        // Create a new store and load from file
        let mut loaded_store = InMemoryVectorStore::new();
        loaded_store.load(&file_path).unwrap();

        // Verify the loaded store has the same data
        assert_eq!(loaded_store.size(), store.size());

        // Check metadata was preserved
        assert!(loaded_store.metadata.is_some());
        let meta = loaded_store.metadata.unwrap();
        assert_eq!(meta.embedding_model, "mock_model_64d");
        assert_eq!(meta.embedding_dimension, 64);
    }
}