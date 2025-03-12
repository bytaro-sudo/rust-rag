//! Example showing how to use embeddings with the rust-rag library.

use rust_rag::chunking::Document;
use rust_rag::chunking::ChunkingStrategy;
use rust_rag::chunking::chunk_document;
use rust_rag::embedding::EmbeddingConfig;
use rust_rag::embedding::EmbeddingProvider;
use rust_rag::embedding::create_embedding_generator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test document
    let doc = Document::new(
        "example-doc-1",
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, \
        and guarantees thread safety. It accomplishes these goals by being memory safe without \
        using garbage collection. Rust is syntactically similar to C++, but can guarantee memory \
        safety by using a borrow checker to validate references."
    )
    .with_metadata("source", "Rust Wikipedia")
    .with_metadata("author", "Rust Community");

    println!("Processing document: {}", doc.id);

    // Chunk the document
    let chunks = chunk_document(&doc, &ChunkingStrategy::Paragraph)?;
    println!("Created {} chunks", chunks.len());

    // Configure a mock embedding generator
    let config = EmbeddingConfig {
        provider: EmbeddingProvider::Mock { dimension: 128 },
    };

    // Create the embedding generator
    let generator = create_embedding_generator(&config)?;
    println!("Using embedding model: {}", generator.model_name());

    // Generate embeddings for all chunks
    let embeddings = generator.embed_chunks(&chunks)?;
    println!("Generated {} embeddings", embeddings.len());

    // Display information about the embeddings
    for (i, embedding) in embeddings.iter().enumerate() {
        println!(
            "Embedding {}: dimension={}, chunk_id={}",
            i,
            embedding.dimension,
            embedding.chunk_id
        );

        // Print first few vector values
        print!("Vector preview: [");
        for j in 0..5.min(embedding.vector.len())
        {
            print!("{:.4}, ", embedding.vector[j]);
        }
        println!("...]");
    }

    //Demonstrate similarity calculation
    if embeddings.len() >= 2 {
        let similarity = embeddings[0].cosine_similarity(&embeddings[1]);
        println!("Similarity between first two chunks: {:.4}", similarity);
    }

    Ok(())
}