// tests/integration_test.rs

use rust_rag::{
    chunking::{ChunkingStrategy, Document, chunk_document},
    embedding::{EmbeddingConfig, EmbeddingProvider, create_embedding_generator},
    retrieval::{create_retriever},
    storage::{VectorStore, InMemoryVectorStore},
    generation::{GenerationConfig, LLMProvider, create_text_generator},
};
use std::collections::HashMap;

// Basic test for chunking + embedding flow
#[test]
fn test_chunking_and_embedding_integration() {
    // 1. Create a test document
    let doc = Document::new(
        "test-doc-1",
        "This is a test document for the RAG system in Rust. \
         It contains multiple paragraphs to verify the correct functioning \
         of the chunking system.\n\n\
         This is the second paragraph of the document. It should be detected as \
         an independent chunk when we use the paragraph strategy."
    )
        .with_metadata("source", "Integration Test")
        .with_metadata("author", "Test Developer");

    // 2. Chunk the document using paragraph strategy
    let chunks = chunk_document(&doc, &ChunkingStrategy::Paragraph).unwrap();

    // Verify that two chunks were created
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].document_id, "test-doc-1");

    // 3. Configure the mock embedding generator
    let config = EmbeddingConfig {
        provider: EmbeddingProvider::Mock { dimension: 64 },
    };

    let generator = create_embedding_generator(&config).unwrap();

    // 4. Generate embeddings for the chunks
    let embeddings = generator.embed_chunks(&chunks).unwrap();

    // Verify that an embedding was generated for each chunk
    assert_eq!(embeddings.len(), chunks.len());
    assert_eq!(embeddings[0].dimension, 64);

    // 5. Verify that we can calculate similarity between embeddings
    let similarity = embeddings[0].cosine_similarity(&embeddings[1]);

    // Similarity should be a value between 0 and 1
    assert!(similarity >= 0.0 && similarity <= 1.0);
}

// Test fixed-size chunking strategy
#[test]
fn test_fixed_size_chunking() {
    let text = "This is a long text that should be divided into multiple chunks when \
               using the fixed size strategy. The chosen size is small enough \
               to force multiple divisions.";

    let doc = Document::new("fixed-size-test", text);

    // Configure a small chunk size to force multiple chunks
    let chunks = chunk_document(&doc, &ChunkingStrategy::FixedSize(30)).unwrap();

    // Multiple chunks should be generated
    assert!(chunks.len() > 1);

    // Verify that no chunk exceeds the maximum size
    // (can be slightly larger due to the word boundary algorithm)
    for chunk in &chunks {
        // Allow some margin due to the space-seeking behavior
        assert!(chunk.content.len() <= 40);
    }

    // Verify that the chunks contain all the original text
    let reconstructed = chunks.iter()
        .map(|c| c.content.clone())
        .collect::<Vec<String>>()
        .join(" ");

    // The reconstruction should contain the same words (ignoring extra spaces)
    let original_words: Vec<&str> = text.split_whitespace().collect();
    let reconstructed_words: Vec<&str> = reconstructed.split_whitespace().collect();

    assert_eq!(original_words, reconstructed_words);
}

// Test sentence-based chunking strategy
#[test]
fn test_sentence_chunking() {
    let text = "This is the first sentence. This is the second sentence. And this is the third.";
    let doc = Document::new("sentence-test", text);

    let chunks = chunk_document(&doc, &ChunkingStrategy::Sentence).unwrap();

    // Three chunks should be generated (one per sentence)
    assert_eq!(chunks.len(), 3);

    // Verify that each chunk contains a complete sentence
    assert!(chunks[0].content.contains("first sentence"));
    assert!(chunks[1].content.contains("second sentence"));
    assert!(chunks[2].content.contains("third"));
}

// Test metadata preservation in chunks
#[test]
fn test_metadata_preservation() {
    let mut metadata = HashMap::new();
    metadata.insert("author".to_string(), "Test Author".to_string());
    metadata.insert("date".to_string(), "2025-03-12".to_string());

    let doc = Document {
        id: "metadata-test".to_string(),
        content: "Test content to verify metadata preservation.".to_string(),
        metadata,
    };

    let chunks = chunk_document(&doc, &ChunkingStrategy::Paragraph).unwrap();

    // Verify that metadata is preserved in the chunks
    assert_eq!(chunks[0].metadata.get("author").unwrap(), "Test Author");
    assert_eq!(chunks[0].metadata.get("date").unwrap(), "2025-03-12");
}

// Test that chunk IDs are generated correctly
#[test]
fn test_chunk_id_generation() {
    let doc = Document::new("id-test", "Content for ID testing");

    let chunks = chunk_document(&doc, &ChunkingStrategy::Paragraph).unwrap();

    // Verify that the chunk ID includes the document ID and a position number
    assert_eq!(chunks[0].id, "id-test_0");

    // If we generate more chunks, they should have sequential IDs
    let doc2 = Document::new("id-test", "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.");
    let chunks2 = chunk_document(&doc2, &ChunkingStrategy::Paragraph).unwrap();

    assert_eq!(chunks2[0].id, "id-test_0");
    assert_eq!(chunks2[1].id, "id-test_1");
    assert_eq!(chunks2[2].id, "id-test_2");
}

// Test the complete RAG pipeline
#[test]
fn test_complete_rag_pipeline() {
    // 1. Create documents
    let doc = Document::new(
        "test-rag-doc",
        "Rust is a systems programming language that ensures memory safety without using garbage collection.\n\n\
         The ownership system in Rust is unique and helps prevent many common bugs found in other languages.\n\n\
         Rust has been voted the most loved programming language for several years in a row."
    );

    // 2. Create embedding generator
    let config = EmbeddingConfig {
        provider: EmbeddingProvider::Mock { dimension: 64 },
    };
    let embedding_generator = create_embedding_generator(&config).unwrap();

    // 3. Chunk the document
    let chunks = chunk_document(&doc, &ChunkingStrategy::Paragraph).unwrap();
    assert_eq!(chunks.len(), 3);

    // 4. Generate embeddings
    let embeddings = embedding_generator.embed_chunks(&chunks).unwrap();
    assert_eq!(embeddings.len(), 3);

    // 5. Create retriever
    let retriever = create_retriever(
        chunks.clone(),
        Some(embeddings),
        Some(embedding_generator.as_ref()),
    ).unwrap();

    // 6. Test retrieval
    let query = "What makes Rust memory safe?";
    let results = retriever.retrieve(query, 2).unwrap();

    // Should retrieve at least one result
    assert!(!results.is_empty());

    // 7. Create text generator
    let gen_config = GenerationConfig {
        provider: LLMProvider::Mock,
        max_tokens: 100,
        temperature: 0.7,
        include_sources: true,
    };
    let text_generator = create_text_generator(&gen_config).unwrap();

    // 8. Generate response
    let generation_result = text_generator.generate(query, &results, &gen_config).unwrap();

    // Verify the response contains relevant information
    assert!(!generation_result.response.is_empty());
    assert!(generation_result.context.is_some());
    assert_eq!(generation_result.context.unwrap().len(), results.len());
}

// Test the storage and retrieval workflow
#[test]
fn test_storage_and_retrieval_workflow() {
    // 1. Create documents
    let doc = Document::new(
        "storage-test-doc",
        "This is a test document for the storage system.\n\n\
         It contains multiple paragraphs that will be stored in the vector store."
    );

    // 2. Create embedding generator
    let config = EmbeddingConfig {
        provider: EmbeddingProvider::Mock { dimension: 64 },
    };
    let generator = create_embedding_generator(&config).unwrap();

    // 3. Chunk the document and generate embeddings
    let chunks = chunk_document(&doc, &ChunkingStrategy::Paragraph).unwrap();
    let embeddings = generator.embed_chunks(&chunks).unwrap();

    // 4. Create vector store
    let mut store = InMemoryVectorStore::new().with_metadata(generator.model_name(), 64);

    // 5. Store chunks and embeddings
    store.add_many(embeddings.clone(), chunks.clone()).unwrap();

    // 6. Verify store size
    assert_eq!(store.size(), chunks.len());

    // 7. Create a query embedding
    let query = "test document";
    let query_embedding = generator.generate(query).unwrap();

    // 8. Search the store
    let results = store.search(&query_embedding, 2).unwrap();

    // Should find results
    assert!(!results.is_empty());

    // 9. Verify the retrieved chunks match the original
    let retrieved_ids: Vec<&String> = results.iter().map(|(chunk, _)| &chunk.id).collect();
    let original_ids: Vec<&String> = chunks.iter().map(|chunk| &chunk.id).collect();

    // At least one of the original chunks should be retrieved
    assert!(retrieved_ids.iter().any(|id| original_ids.contains(id)));
}