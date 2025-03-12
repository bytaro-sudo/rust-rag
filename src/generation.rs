//! Module for integrating with LLMs to generate responses.

use crate::retrieval::RetrievalResult;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::fmt;

/// Configuration for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// The LLM provider to use
    pub provider: LLMProvider,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for generation (higher = more creative)
    pub temperature: f32,
    /// Whether to include retrieved contexts in the response
    pub include_sources: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            provider: LLMProvider::Mock,
            max_tokens: 512,
            temperature: 0.7,
            include_sources: true,
        }
    }
}

/// Available LLM providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMProvider {
    /// OpenAI API (ChatGPT, GPT-4, etc.)
    OpenAI {
        api_key: String,
        model_name: String,
    },
    /// Anthropic API (Claude, etc.)
    Anthropic {
        api_key: String,
        model_name: String,
    },
    /// Local model using Ollama
    Ollama {
        host: String,
        model_name: String,
    },
    /// Mock provider for testing
    Mock,
}

impl fmt::Display for LLMProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMProvider::OpenAI { model_name, .. } => write!(f, "OpenAI ({})", model_name),
            LLMProvider::Anthropic { model_name, .. } => write!(f, "Anthropic ({})", model_name),
            LLMProvider::Ollama { model_name, .. } => write!(f, "Ollama ({})", model_name),
            LLMProvider::Mock => write!(f, "Mock LLM"),
        }
    }
}

/// Result from the generation process.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The generated text response
    pub response: String,
    /// The context used to generate the response (if include_sources is true)
    pub context: Option<Vec<RetrievalSource>>,
    /// The model used for generation
    pub model: String,
    /// Total tokens used (prompt + completion)
    pub tokens_used: usize,
}

/// Information about a source used in generation.
#[derive(Debug, Clone)]
pub struct RetrievalSource {
    /// ID of the document
    pub document_id: String,
    /// ID of the chunk
    pub chunk_id: String,
    /// Content of the chunk
    pub content: String,
    /// Relevance score of the chunk
    pub score: f32,
    /// Metadata associated with the chunk
    pub metadata: std::collections::HashMap<String, String>,
}

impl From<RetrievalResult> for RetrievalSource {
    fn from(result: RetrievalResult) -> Self {
        Self {
            document_id: result.chunk.document_id,
            chunk_id: result.chunk.id,
            content: result.chunk.content,
            score: result.score,
            metadata: result.chunk.metadata,
        }
    }
}

/// Interface for text generators.
pub trait TextGenerator {
    /// Generate text based on a query and retrieved contexts.
    fn generate(&self, query: &str, contexts: &[RetrievalResult], config: &GenerationConfig) -> Result<GenerationResult>;

    /// Get the model name used by this generator.
    fn model_name(&self) -> &str;
}

/// A mock text generator for testing.
pub struct MockTextGenerator {
    model_name: String,
}

impl MockTextGenerator {
    /// Create a new mock text generator.
    pub fn new() -> Self {
        Self {
            model_name: "mock-llm-model".to_string(),
        }
    }
}

impl Default for MockTextGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGenerator for MockTextGenerator {
    fn generate(&self, query: &str, contexts: &[RetrievalResult], config: &GenerationConfig) -> Result<GenerationResult> {
        // Create a simple response that includes the query and summarizes the contexts
        let mut response = format!("Query: {}\n\nResponse: ", query);

        if contexts.is_empty() {
            response.push_str("No relevant information found for this query.");
        } else {
            response.push_str("Based on the retrieved information, I can provide the following answer:\n\n");

            // Summarize the contexts
            let contexts_summary: Vec<String> = contexts
                .iter()
                .map(|result| format!("- {}", result.chunk.content))
                .collect();

            response.push_str(&contexts_summary.join("\n"));
        }

        // Calculate mock token usage
        let tokens_used = query.split_whitespace().count() + response.split_whitespace().count();

        // Create the result
        let result = GenerationResult {
            response,
            context: if config.include_sources {
                Some(contexts.iter().cloned().map(Into::into).collect())
            } else {
                None
            },
            model: self.model_name.clone(),
            tokens_used,
        };

        Ok(result)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// OpenAI text generator.
#[cfg(feature = "openai")]
pub struct OpenAIGenerator {
    api_key: String,
    model_name: String,
    client: reqwest::Client,
}

#[cfg(feature = "openai")]
impl OpenAIGenerator {
    /// Create a new OpenAI text generator.
    pub fn new(api_key: String, model_name: String) -> Self {
        Self {
            api_key,
            model_name,
            client: reqwest::Client::new(),
        }
    }

    /// Format the prompt for the API.
    fn format_prompt(&self, query: &str, contexts: &[RetrievalResult]) -> String {
        let mut prompt = String::new();

        // Add contexts if available
        if !contexts.is_empty() {
            prompt.push_str("Context information:\n");

            for (i, result) in contexts.iter().enumerate() {
                prompt.push_str(&format!("[{i}] {}\n\n", result.chunk.content));
            }

            prompt.push_str("\nGiven the context information and not prior knowledge, answer the query.\n\n");
        }

        // Add the query
        prompt.push_str(&format!("Query: {query}\n\nAnswer:"));

        prompt
    }
}

#[cfg(feature = "openai")]
impl TextGenerator for OpenAIGenerator {
    fn generate(&self, query: &str, contexts: &[RetrievalResult], config: &GenerationConfig) -> Result<GenerationResult> {
        // Implementation would use the OpenAI API to generate a response
        unimplemented!("OpenAI integration requires the 'openai' feature")
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Factory function to create a text generator based on configuration.
pub fn create_text_generator(config: &GenerationConfig) -> Result<Box<dyn TextGenerator>> {
    match &config.provider {
        LLMProvider::Mock => {
            Ok(Box::new(MockTextGenerator::new()))
        },
        #[cfg(feature = "openai")]
        LLMProvider::OpenAI { api_key, model_name } => {
            Ok(Box::new(OpenAIGenerator::new(api_key.clone(), model_name.clone())))
        },
        _ => Err(anyhow!("LLM provider not implemented yet: {}", config.provider)),
    }
}

/// Prompt templates for different LLM providers.
pub struct PromptTemplates;

impl PromptTemplates {
    /// Default RAG prompt template.
    pub fn default_rag_template(query: &str, contexts: &[RetrievalResult]) -> String {
        let mut prompt = String::new();

        // Add contexts
        if !contexts.is_empty() {
            prompt.push_str("You are an AI assistant answering a question based on the provided context information below.\n");
            prompt.push_str("Context information:\n");

            for (i, result) in contexts.iter().enumerate() {
                prompt.push_str(&format!("--- Context {i} ---\n{}\n\n", result.chunk.content));
            }

            prompt.push_str("Instructions: Based solely on the context provided above, answer the following query. If the context doesn't contain relevant information, say 'I don't have enough information to answer this question.'\n\n");
        } else {
            prompt.push_str("You are an AI assistant answering a question. No specific context information is provided.\n\n");
        }

        // Add the query
        prompt.push_str(&format!("Query: {query}\n\nAnswer:"));

        prompt
    }

    /// OpenAI-specific template.
    #[cfg(feature = "openai")]
    pub fn openai_template(query: &str, contexts: &[RetrievalResult]) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();

        // System message
        messages.push(serde_json::json!({
            "role": "system",
            "content": "You are a helpful assistant answering questions based on provided context information."
        }));

        // Add contexts
        if !contexts.is_empty() {
            let context_str = contexts.iter()
                .enumerate()
                .map(|(i, result)| format!("--- Context {i} ---\n{}", result.chunk.content))
                .collect::<Vec<_>>()
                .join("\n\n");

            messages.push(serde_json::json!({
                "role": "user",
                "content": format!("Here is context information to help answer a question:\n\n{context_str}\n\nUse this context to answer my next question. Only use the provided information.")
            }));

            messages.push(serde_json::json!({
                "role": "assistant",
                "content": "I'll answer your question based on the context you've provided."
            }));
        }

        // Add the query
        messages.push(serde_json::json!({
            "role": "user",
            "content": query
        }));

        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::{Chunk, Document};

    fn create_test_retrieval_results() -> Vec<RetrievalResult> {
        let chunks = vec![
            Chunk {
                id: "chunk1".to_string(),
                content: "Rust is a systems programming language that emphasizes safety and performance.".to_string(),
                document_id: "doc1".to_string(),
                position: 0,
                metadata: Default::default(),
            },
            Chunk {
                id: "chunk2".to_string(),
                content: "Rust's memory safety guarantees are enforced at compile time.".to_string(),
                document_id: "doc1".to_string(),
                position: 1,
                metadata: Default::default(),
            },
        ];

        vec![
            RetrievalResult {
                chunk: chunks[0].clone(),
                score: 0.9,
            },
            RetrievalResult {
                chunk: chunks[1].clone(),
                score: 0.8,
            },
        ]
    }

    #[test]
    fn test_mock_text_generator() {
        let generator = MockTextGenerator::new();
        let config = GenerationConfig::default();
        let contexts = create_test_retrieval_results();

        let result = generator.generate("What is Rust?", &contexts, &config).unwrap();

        assert!(result.response.contains("What is Rust?"));
        assert!(result.response.contains("systems programming language"));
        assert!(result.context.is_some());
        assert_eq!(result.context.unwrap().len(), 2);
    }

    #[test]
    fn test_prompt_templates() {
        let contexts = create_test_retrieval_results();
        let query = "What is Rust?";

        let prompt = PromptTemplates::default_rag_template(query, &contexts);

        assert!(prompt.contains("Context information:"));
        assert!(prompt.contains("systems programming language"));
        assert!(prompt.contains("Query: What is Rust?"));
    }
}