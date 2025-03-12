//! Module for indexing documents for efficient retrieval.

use crate::chunking::{Document, Chunk, ChunkingStrategy, chunk_document};
use crate::embedding::{EmbeddingGenerator};
use crate::storage::VectorStore;
use anyhow::{Result, Context};
use std::sync::Arc;
use log::{info, warn, debug};

/// Result of an indexing operation.
#[derive(Debug)]
pub struct IndexingResult {
    /// Document ID that was indexed
    pub document_id: String,
    /// Number of chunks created
    pub chunk_count: usize,
    /// Number of embeddings generated
    pub embedding_count: usize,
}

/// Configuration for the indexing process.
#[derive(Debug, Clone)]
pub struct IndexingConfig {
    /// The chunking strategy to use
    pub chunking_strategy: ChunkingStrategy,
    /// Maximum number of chunks to process at once
    pub batch_size: usize,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            chunking_strategy: ChunkingStrategy::Paragraph,
            batch_size: 10,
        }
    }
}

/// Service for indexing documents.
pub struct IndexingService<G, S>
where
    G: EmbeddingGenerator,
    S: VectorStore,
{
    /// The embedding generator
    embedding_generator: Arc<G>,
    /// The vector store
    store: S,
    /// Configuration for indexing
    config: IndexingConfig,
}

impl<G, S> IndexingService<G, S>
where
    G: EmbeddingGenerator,
    S: VectorStore,
{
    /// Create a new indexing service.
    pub fn new(
        embedding_generator: Arc<G>,
        store: S,
        config: IndexingConfig,
    ) -> Self {
        Self {
            embedding_generator,
            store,
            config,
        }
    }

    /// Index a single document.
    pub fn index_document(&mut self, doc: Document) -> Result<IndexingResult> {
        debug!("Indexing document: {}", doc.id);

        // Step 1: Chunk the document
        let chunks = chunk_document(&doc, &self.config.chunking_strategy)
            .context("Failed to chunk document")?;

        debug!("Created {} chunks for document {}", chunks.len(), doc.id);

        // Step 2: Process chunks in batches to avoid memory issues with large documents
        let mut processed_chunks = 0;
        let total_chunks = chunks.len();

        for chunk_batch in chunks.chunks(self.config.batch_size) {
            // Create a batch of chunks
            let batch: Vec<Chunk> = chunk_batch.to_vec();

            // Generate embeddings for the batch
            let embeddings = self.embedding_generator.embed_chunks(&batch)
                .context("Failed to generate embeddings for chunks")?;

            // Store embeddings and chunks
            self.store.add_many(embeddings.clone(), batch.clone())
                .context("Failed to store embeddings and chunks")?;

            processed_chunks += batch.len();
            debug!("Processed {}/{} chunks", processed_chunks, total_chunks);
        }

        info!("Successfully indexed document {} with {} chunks", doc.id, total_chunks);

        Ok(IndexingResult {
            document_id: doc.id,
            chunk_count: total_chunks,
            embedding_count: total_chunks, // Assuming one embedding per chunk
        })
    }

    /// Index multiple documents.
    pub fn index_documents(&mut self, docs: Vec<Document>) -> Result<Vec<IndexingResult>> {
        let mut results = Vec::new();

        for doc in docs {
            match self.index_document(doc) {
                Ok(result) => results.push(result),
                Err(e) => {
                    warn!("Failed to index document: {}", e);
                    // Continue with other documents even if one fails
                }
            }
        }

        info!("Indexed {} documents successfully", results.len());

        Ok(results)
    }

    /// Get the current size of the index.
    pub fn index_size(&self) -> usize {
        self.store.size()
    }

    /// Get reference to the store.
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get mutable reference to the store.
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{EmbeddingConfig, EmbeddingProvider, create_embedding_generator, MockEmbeddingGenerator};
    use crate::storage::InMemoryVectorStore;

    #[test]
    fn test_indexing_service() {
        // Create a mock embedding generator
        let generator = MockEmbeddingGenerator::new(64);
        let generator = Arc::new(generator);

        // Create an in-memory store
        let store = InMemoryVectorStore::new().with_metadata(generator.model_name(), 64);

        // Create the indexing service
        let config = IndexingConfig::default();
        let mut service = IndexingService::new(generator, store, config);

        // Create a test document
        let doc = Document::new(
            "test-doc-1",
            "This is a test document. It has multiple sentences.\n\nThis is a second paragraph."
        )
            .with_metadata("source", "test");

        // Index the document
        let result = service.index_document(doc).unwrap();

        // Verify the result
        assert_eq!(result.document_id, "test-doc-1");
        assert_eq!(result.chunk_count, 2); // Two paragraphs
        assert_eq!(result.embedding_count, 2);

        // Verify store size
        assert_eq!(service.index_size(), 2);
    }

    #[test]
    fn test_batch_processing() {
        // Create a mock embedding generator
        let generator = MockEmbeddingGenerator::new(64);
        let generator = Arc::new(generator);

        // Create an in-memory store
        let store = InMemoryVectorStore::new().with_metadata(generator.model_name(), 64);

        // Create the indexing service with small batch size
        let config = IndexingConfig {
            chunking_strategy: ChunkingStrategy::Sentence,
            batch_size: 2, // Small batch size to test batching
        };
        let mut service = IndexingService::new(generator, store, config);

        // Create a test document with multiple sentences
        let doc = Document::new(
            "test-doc-2",
            "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        );

        // Index the document
        let result = service.index_document(doc).unwrap();

        // Verify the result
        assert_eq!(result.chunk_count, 4); // Four sentences
        assert_eq!(service.index_size(), 4);
    }
}