//! A simple example of using the rust-rag library.

use rust_rag::{
    chunking::{Document, ChunkingStrategy, chunk_document},
    embedding::{EmbeddingConfig, EmbeddingProvider, create_embedding_generator},
    retrieval::{create_retriever, Retriever},
    generation::{GenerationConfig, LLMProvider, create_text_generator},
    storage::InMemoryVectorStore,
};
use std::error::Error;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Rust RAG Example");
    println!("----------------");

    // Step 1: Create sample documents
    println!("\n1. Creating sample documents...");
    let documents = create_sample_documents();
    println!("   Created {} documents", documents.len());

    // Step 2: Create embedding generator
    println!("\n2. Setting up embedding generator...");
    let config = EmbeddingConfig {
        provider: EmbeddingProvider::Mock { dimension: 128 },
    };
    let embedding_generator = create_embedding_generator(&config)?;
    println!("   Using embedding model: {}", embedding_generator.model_name());

    // Step 3: Process documents (chunking + embedding)
    println!("\n3. Processing documents...");

    // Store all chunks and embeddings
    let mut all_chunks = Vec::new();
    let mut all_embeddings = Vec::new();

    for doc in &documents {
        println!("   Processing document: {}", doc.id);

        // Chunk the document
        let chunks = chunk_document(doc, &ChunkingStrategy::Paragraph)?;
        println!("     Created {} chunks", chunks.len());

        // Generate embeddings
        let embeddings = embedding_generator.embed_chunks(&chunks)?;
        println!("     Generated {} embeddings", embeddings.len());

        // Store for later use
        all_chunks.extend(chunks);
        all_embeddings.extend(embeddings);
    }

    // Step 4: Create retriever
    println!("\n4. Setting up retriever...");
    let retriever = create_retriever(
        all_chunks,
        Some(all_embeddings),
        Some(embedding_generator.as_ref()),
    )?;
    println!("   Retriever ready");

    // Step 5: Create text generator
    println!("\n5. Setting up text generator...");
    let gen_config = GenerationConfig {
        provider: LLMProvider::Mock,
        max_tokens: 256,
        temperature: 0.7,
        include_sources: true,
    };
    let text_generator = create_text_generator(&gen_config)?;
    println!("   Using generator model: {}", text_generator.model_name());

    // Step 6: Answer queries
    println!("\n6. Ready to answer queries!");

    // List of sample queries
    let queries = vec![
        "What are the key features of Rust?",
        "How does Rust ensure memory safety?",
        "What is the ownership model?",
    ];

    for (i, query) in queries.iter().enumerate() {
        println!("\n----- Query {} -----", i + 1);
        println!("Q: {}", query);

        // Retrieve relevant chunks
        let retrieved = retriever.retrieve(query, 3)?;
        println!("Retrieved {} relevant chunks", retrieved.len());

        // Generate response
        let result = text_generator.generate(query, &retrieved, &gen_config)?;

        // Display the response
        println!("\nA: {}", result.response);

        // Display source information if available
        if let Some(sources) = result.context {
            println!("\nSources:");
            for (i, source) in sources.iter().enumerate() {
                println!("  [{}] Document: {}, Chunk: {}, Score: {:.2}",
                         i + 1, source.document_id, source.chunk_id, source.score);
            }
        }

        println!("\nTokens used: {}", result.tokens_used);
    }

    Ok(())
}

/// Create sample documents for the example.
fn create_sample_documents() -> Vec<Document> {
    vec![
        Document::new(
            "rust-intro",
            "Rust is a systems programming language that emphasizes safety, concurrency, and performance. \
            It was initially designed by Graydon Hoare at Mozilla Research, with contributions from Dave Herman, \
            Brendan Eich, and others.\n\n\
            Rust has been voted the 'most loved programming language' in the Stack Overflow Developer Survey \
            since 2016. It is primarily used for applications where safety, performance, and concurrent operation \
            are important."
        )
            .with_metadata("source", "Rust Documentation")
            .with_metadata("date", "2023-01-15"),

        Document::new(
            "rust-memory-safety",
            "Rust's memory safety guarantees are enforced at compile time through a system of ownership, \
            borrowing, and lifetimes. This approach eliminates entire classes of bugs that are common in other \
            systems programming languages like C and C++, such as buffer overflows, dangling pointers, and data races.\n\n\
            The ownership system in Rust tracks who can read and write to memory, ensures that pointers are always valid, \
            and guarantees that concurrent code is free from data races. These checks happen at compile time, with minimal \
            runtime overhead."
        )
            .with_metadata("source", "Rust Memory Safety Guide")
            .with_metadata("date", "2023-02-20"),

        Document::new(
            "rust-concurrency",
            "Rust provides tools for fearless concurrency. The ownership and type systems ensure thread safety \
            at compile time. Rust lets you write code that's free of subtle bugs and is easy to refactor without \
            introducing new bugs.\n\n\
            Concurrency in Rust is handled through various abstractions, including threads, async/await, and \
            message-passing channels. The type system and borrow checker enforce rules that prevent data races \
            and other concurrency issues."
        )
            .with_metadata("source", "Rust Concurrency Patterns")
            .with_metadata("date", "2023-03-10"),
    ]
}