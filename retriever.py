"""Retriever function for querying the vector database."""

from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, SparseVector, NamedVector, NamedSparseVector
import torch
import logging

from config import settings
from models import RetrievedChunk

# Configure logger
logger = logging.getLogger(__name__)

# Import fastembed for sparse vectors
try:
    from fastembed import SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


class Retriever:
    """Handles retrieval of relevant chunks from the vector database."""
    
    def __init__(self):
        """Initialize the retriever with embedding model and Qdrant client."""
        # Detect GPU availability for all models
        if settings.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🎮 GPU detected: {gpu_name}")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = 'cpu'
            if not settings.use_gpu:
                logger.info(f"💻 GPU disabled in config, using CPU")
            else:
                logger.info(f"💻 No GPU detected, using CPU")
        
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.embedding_model = SentenceTransformer(settings.embedding_model, device=self.device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"✓ Embedding model loaded: {settings.embedding_model} ({self.embedding_dim} dimensions) on {self.device.upper()}")
        
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        
        # Verify collection vector dimensions match
        try:
            collection_info = self.client.get_collection(settings.qdrant_collection_name)
            if hasattr(collection_info.config.params, 'vectors'):
                vectors_config = collection_info.config.params.vectors
                if isinstance(vectors_config, dict) and 'dense' in vectors_config:
                    collection_dim = vectors_config['dense'].size
                else:
                    # Old format: single vector
                    collection_dim = vectors_config.size if hasattr(vectors_config, 'size') else None
                
                if collection_dim and collection_dim != self.embedding_dim:
                    raise ValueError(
                        f"❌ Dimension mismatch!\n"
                        f"   Collection expects: {collection_dim} dimensions\n"
                        f"   Model produces: {self.embedding_dim} dimensions\n"
                        f"   Model: {settings.embedding_model}\n\n"
                        f"   Solution: Either:\n"
                        f"   1. Change embedding_model in config.py to match collection ({collection_dim} dims), OR\n"
                        f"   2. Delete collection and re-ingest with current model ({self.embedding_dim} dims)"
                    )
                logger.info(f"✓ Collection vector dimensions match: {collection_dim}")
        except Exception as e:
            if "Dimension mismatch" in str(e):
                raise
            logger.warning(f"⚠ Could not verify collection dimensions: {e}")
        
        # Initialize sparse model for hybrid search
        self.sparse_model = None
        self.use_hybrid = settings.use_hybrid_search and FASTEMBED_AVAILABLE
        if self.use_hybrid:
            try:
                self.sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
                logger.info(f"✓ Retriever initialized with hybrid search (Dense + Sparse BM25)")
            except Exception as e:
                logger.warning(f"⚠ Failed to load sparse model: {e}")
                logger.info("  Falling back to dense-only retrieval")
                self.use_hybrid = False
        else:
            logger.info(f"✓ Retriever initialized with dense-only search")
        
        # Initialize reranker model
        self.reranker = None
        self.use_reranking = settings.use_reranking
        if self.use_reranking:
            try:
                logger.info(f"Loading reranker model: {settings.reranker_model}")
                self.reranker = CrossEncoder(settings.reranker_model, device=self.device)
                logger.info(f"✓ Reranker loaded: {settings.reranker_model} on {self.device.upper()}")
            except Exception as e:
                logger.warning(f"⚠ Failed to load reranker: {e}")
                logger.info("  Falling back to retrieval without reranking")
                self.use_reranking = False
    
    def retrieve(
        self,
        question: str,
        k: int = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        modality_tags: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a given question.
        
        Args:
            question: The question to search for
            k: Number of top results to return (default from settings)
            year_min: Minimum publication year filter
            year_max: Maximum publication year filter
            modality_tags: List of modality tags to filter by (OR condition)
            paper_ids: List of specific paper IDs to filter by
        
        Returns:
            List of RetrievedChunk objects with relevance scores
        """
        k = k or settings.default_top_k
        
        # Build filters
        filter_conditions = []
        
        # Year range filter
        if year_min is not None or year_max is not None:
            filter_conditions.append(
                FieldCondition(
                    key="year",
                    range=Range(
                        gte=year_min,
                        lte=year_max
                    )
                )
            )
        
        # Modality tags filter (match any of the provided tags)
        if modality_tags:
            for tag in modality_tags:
                filter_conditions.append(
                    FieldCondition(
                        key="modality_tags",
                        match=MatchValue(value=tag)
                    )
                )
        
        # Paper IDs filter
        if paper_ids:
            for paper_id in paper_ids:
                filter_conditions.append(
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id)
                    )
                )
        
        # Construct the filter object
        query_filter = None
        if filter_conditions:
            query_filter = Filter(should=filter_conditions)  # OR condition
        
        # Query Qdrant (hybrid or dense-only)
        if self.use_hybrid and self.sparse_model:
            # Hybrid search: Dense + Sparse BM25
            dense_embedding = self.embedding_model.encode(question)
            sparse_emb = list(self.sparse_model.embed([question]))[0]
            
            # Convert FastEmbed SparseEmbedding to Qdrant SparseVector
            sparse_vector = SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist()
            )
            
            # Perform separate searches for dense and sparse vectors
            # For named vectors, use NamedVector and NamedSparseVector objects
            try:
                # Search dense vectors using NamedVector
                dense_query_vector = NamedVector(
                    name="dense",
                    vector=dense_embedding.tolist()
                )
                dense_results = self.client.search(
                    collection_name=settings.qdrant_collection_name,
                    query_vector=dense_query_vector,
                    query_filter=query_filter,
                    limit=k * 2,
                    with_payload=True
                )
                
                # Search sparse vectors using NamedSparseVector
                sparse_query_vector = NamedSparseVector(
                    name="sparse",
                    vector=sparse_vector
                )
                sparse_results = self.client.search(
                    collection_name=settings.qdrant_collection_name,
                    query_vector=sparse_query_vector,
                    query_filter=query_filter,
                    limit=k * 2,
                    with_payload=True
                )
            except (TypeError, ValueError, AttributeError) as e:
                # Fallback: If named vector format doesn't work, try without names
                logger.warning(f"⚠ Warning: Named vector search failed ({e}), trying fallback...")
                try:
                    dense_results = self.client.search(
                        collection_name=settings.qdrant_collection_name,
                        query_vector=dense_embedding.tolist(),
                        query_filter=query_filter,
                        limit=k * 2,
                        with_payload=True
                    )
                except Exception:
                    # If even fallback fails, raise the original error
                    raise ValueError(
                        f"Failed to search collection. Collection requires named vectors (dense, sparse). "
                        f"Error: {e}"
                    )
                # For sparse, we can't use regular search, so just use dense
                sparse_results = []
            
            # Manual RRF (Reciprocal Rank Fusion)
            from collections import defaultdict
            rrf_scores = defaultdict(float)
            all_results = {}
            
            # Calculate RRF scores from dense results
            for rank, result in enumerate(dense_results, 1):
                rrf_scores[result.id] += 1.0 / (60 + rank)
                all_results[result.id] = result
            
            # Calculate RRF scores from sparse results
            for rank, result in enumerate(sparse_results, 1):
                rrf_scores[result.id] += 1.0 / (60 + rank)
                all_results[result.id] = result
            
            # Sort by RRF score and take top k
            sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_ids = [result_id for result_id, _ in sorted_by_rrf[:k]]
            
            # Get results in RRF order
            search_results = [all_results[result_id] for result_id in top_k_ids if result_id in all_results]
            
            # Update scores to RRF scores
            for result in search_results:
                result.score = rrf_scores.get(result.id, 0.0)
        else:
            # Dense-only search (backward compatible)
            question_embedding = self.embedding_model.encode(question)
            
            search_results = self.client.search(
                collection_name=settings.qdrant_collection_name,
                query_vector=question_embedding.tolist(),
                query_filter=query_filter,
                limit=k,
                with_payload=True
            )
        
        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for result in search_results:
            payload = result.payload
            chunk = RetrievedChunk(
                text=payload["text"],
                title=payload["title"],
                citation_pointer=payload["citation_pointer"],
                section_name=payload["section_name"],
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                paper_id=payload["paper_id"],
                year=payload["year"],
                score=result.score,
                modality_tags=payload.get("modality_tags", [])
            )
            retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def retrieve_with_reranking(
        self,
        question: str,
        k: int = None,
        initial_k: int = None,
        **filter_kwargs
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks with cross-encoder reranking for enhanced quality.
        
        This method performs a two-stage retrieval:
        1. Initial retrieval: Fetch more candidates (initial_k) using vector similarity
        2. Reranking: Use a cross-encoder to rerank the candidates based on semantic relevance
        
        Args:
            question: The question to search for
            k: Final number of results to return (default: settings.rerank_top_k)
            initial_k: Number of initial results before reranking (default: settings.rerank_initial_k)
            **filter_kwargs: Additional filter arguments for retrieve()
        
        Returns:
            List of RetrievedChunk objects, reranked by relevance
        """
        # Use configuration defaults if not provided
        final_k = k or settings.rerank_top_k
        fetch_k = initial_k or settings.rerank_initial_k
        
        # Ensure we fetch enough candidates
        fetch_k = max(fetch_k, final_k)
        
        # Step 1: Initial retrieval with larger k
        chunks = self.retrieve(question=question, k=fetch_k, **filter_kwargs)
        
        if not chunks:
            return []
        
        # If no reranker is available or not enough chunks, return as-is
        if not self.use_reranking or not self.reranker or len(chunks) <= 1:
            return chunks[:final_k]
        
        # Step 2: Rerank using cross-encoder
        try:
            # Prepare pairs of [question, chunk_text] for the cross-encoder
            pairs = [[question, chunk.text] for chunk in chunks]
            
            # Get reranking scores from cross-encoder
            rerank_scores = self.reranker.predict(pairs)
            
            # Update chunk scores with reranking scores and sort
            for chunk, score in zip(chunks, rerank_scores):
                chunk.score = float(score)  # Update with reranker score
            
            # Sort by reranker scores (higher is better)
            chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            
            # Return top k after reranking
            return chunks[:final_k]
        
        except Exception as e:
            logger.warning(f"⚠ Reranking failed: {e}")
            logger.info("  Returning original retrieval results")
            return chunks[:final_k]


def retrieve(
    question: str,
    k: int = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function for retrieval (functional interface).
    
    Args:
        question: The question to search for
        k: Number of top results to return
        filters: Dictionary of filters (year_min, year_max, modality_tags, etc.)
    
    Returns:
        List of dictionaries with chunk data
    """
    retriever = Retriever()
    filter_kwargs = filters or {}
    chunks = retriever.retrieve(question=question, k=k, **filter_kwargs)
    
    # Convert to dictionaries
    return [
        {
            "text": chunk.text,
            "title": chunk.title,
            "citation_pointer": chunk.citation_pointer,
            "section_name": chunk.section_name,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "paper_id": chunk.paper_id,
            "year": chunk.year,
            "score": chunk.score,
            "modality_tags": chunk.modality_tags
        }
        for chunk in chunks
    ]


if __name__ == "__main__":
    # Simple test
    retriever = Retriever()
    results = retriever.retrieve(
        question="How does emotion affect gait patterns?",
        k=5
    )
    
    logger.info(f"Found {len(results)} results:")
    for i, chunk in enumerate(results, 1):
        logger.info(f"\n{i}. [{chunk.citation_pointer} | {chunk.section_name}]")
        logger.info(f"   Score: {chunk.score:.4f}")
        logger.info(f"   {chunk.text[:200]}...")


