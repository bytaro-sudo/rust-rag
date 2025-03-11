//! # Rust RAG
//!
//! `rust-rag` is a modular implementation of a Retrieval-Augmented Generation (RAG) system
//! in Rust, designed for high performance and flexibility
pub mod chunking;
pub mod embedding;
pub mod indexing;
pub mod retrieval;
pub mod storage;
pub mod generation;

/// Common re-exports for ease of use
pub use chunking::Document;
pub use chunking::Chunk;
pub use embedding::Embedding;
pub use embedding::EmbeddingGenerator;
pub use retrieval::RetrievalResult;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}