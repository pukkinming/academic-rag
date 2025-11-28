"""Configuration management for the RAG system."""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Vector Database (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "academic_papers"
    
    ###################
    # Embedding Model #
    ###################
    # # original
    # embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # # Option A: Better free model
    embedding_model: str = "BAAI/bge-base-en-v1.5"  # Recommended upgrade!

    # # Option B: Even better free model
    # embedding_model: str = "BAAI/bge-large-en-v1.5"

    # # Option C: Smaller/faster
    # embedding_model: str = "all-MiniLM-L12-v2"

    # Option D: Domain-specific
    # embedding_model: str = "allenai/specter2"  # For scientific papers
    
    ###############################
    # Hybrid Search Configuration #
    ###############################
    use_hybrid_search: bool = True  # Enable dense + sparse (BM25) hybrid search
    # Option A: BM25 (current, recommended for most)
    sparse_model: str = "Qdrant/bm25"

    # # Option B: SPLADE (learned sparse, better but slower)
    # sparse_model: str = "prithivida/Splade_PP_en_v1"

    # # Option C: BM25 variant
    # sparse_model: str = "Qdrant/bm42"  # Qdrant's enhanced version

    # # Option D: learned sparse (best for scientific papers)
    # sparse_model: str = "naver/splade-cocondenser-ensembledistil"
    
    # Hybrid search fusion method: "rrf" (Reciprocal Rank Fusion) or "weighted" (α·dense + (1-α)·sparse)
    hybrid_search_fusion_method: str = "weighted"  # Options: "rrf", "weighted"
    
    # Alpha parameter for weighted fusion: α·dense_score + (1-α)·sparse_score
    # Range: 0.0 to 1.0
    # - 0.0 = pure sparse (BM25 only)
    # - 0.5 = equal weighting (default)
    # - 1.0 = pure dense (semantic only)
    hybrid_search_alpha: float = 0.5
    
    # Embedding Processing Parameters
    embedding_batch_size: int = 4  # Batch size for encoding (reduce if GPU memory issues)
    
    # Device Configuration
    use_gpu: bool = True  # Use GPU if available (set to False to force CPU usage)
    
    # LLM Configuration
    llm_provider: str = "vllm"  # Options: "openai", "vllm"
    # llm_model: str = "gpt-4-turbo-preview"
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None  # For Azure or custom OpenAI endpoints
    
    # Hugging Face Configuration
    hf_token: Optional[str] = None  # Hugging Face token for accessing gated models
    
    @field_validator('openai_base_url', mode='before')
    @classmethod
    def empty_str_to_none(cls, v):
        """Convert empty string to None for openai_base_url."""
        if v == '' or (isinstance(v, str) and not v.strip()):
            return None
        return v
    
    # vLLM Configuration
    # vllm_base_url: str = "http://localhost:8000/v1"  # vLLM OpenAI-compatible endpoint
    vllm_base_url: str = "http://vllm:8000/v1"  # vLLM OpenAI-compatible endpoint
    vllm_api_key: Optional[str] = "EMPTY"  # vLLM typically doesn't require a key
    
    # Chunking Parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval Parameters
    default_top_k: int = 8  # Default number of results; can be overridden per request (max: 100)
    
    #########################
    # Reranking Configuration
    #########################
    use_reranking: bool = True  # Enable cross-encoder reranking for better quality
    rerank_top_k: int = 8  # Final number of results after reranking
    rerank_initial_k: int = 20  # Number of candidates to fetch before reranking (should be > rerank_top_k)
    
    # Reranker model options:
    # Option A: Fast and good for general use (recommended)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # # Option B: Better quality, slightly slower
    # reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # # Option C: Best quality, slowest
    # reranker_model: str = "cross-encoder/ms-marco-TinyBERT-L-6"
    
    # # Option D: Multilingual support
    # reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    
    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
