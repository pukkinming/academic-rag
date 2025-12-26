"""Retriever function for querying the vector database."""

from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, SparseVector
import torch
import torch.nn.functional as F
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
                fusion_method = settings.hybrid_search_fusion_method
                if fusion_method == "weighted":
                    alpha = settings.hybrid_search_alpha
                    logger.info(f"✓ Retriever initialized with hybrid search (Dense + Sparse BM25)")
                    logger.info(f"  Fusion method: Weighted (α={alpha:.2f}·dense + {1-alpha:.2f}·sparse)")
                else:
                    logger.info(f"✓ Retriever initialized with hybrid search (Dense + Sparse BM25)")
                    logger.info(f"  Fusion method: RRF (Reciprocal Rank Fusion)")
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
        
        # Log MMR configuration
        if settings.use_mmr:
            logger.info(f"✓ MMR enabled: λ={settings.mmr_lambda:.2f} (relevance vs diversity)")
            logger.info(f"  Max chunks per paper: {settings.max_chunks_per_paper}")
        elif settings.max_chunks_per_paper < 1000:
            logger.info(f"✓ Per-paper capping enabled: max {settings.max_chunks_per_paper} chunks per paper")
    
    def _apply_mmr_and_capping(
        self,
        chunks: List[RetrievedChunk],
        question: str,
        k: int,
        lambda_param: float = None,
        max_per_paper: int = None
    ) -> List[RetrievedChunk]:
        """
        Apply Maximal Marginal Relevance (MMR) and per-paper chunk capping.
        
        MMR balances relevance to the query with diversity among selected chunks.
        Per-paper capping ensures we don't get too many chunks from the same paper.
        
        Args:
            chunks: List of retrieved chunks (should be sorted by relevance score)
            question: The query question (for computing relevance)
            k: Number of chunks to return
            lambda_param: MMR lambda parameter (0.0 = pure diversity, 1.0 = pure relevance)
            max_per_paper: Maximum chunks per paper (None = no limit)
        
        Returns:
            List of chunks selected using MMR with per-paper capping
        """
        if not chunks:
            return []
        
        # Use settings defaults if not provided
        lambda_param = lambda_param if lambda_param is not None else settings.mmr_lambda
        max_per_paper = max_per_paper if max_per_paper is not None else settings.max_chunks_per_paper
        
        # If MMR is disabled, just apply per-paper capping
        if not settings.use_mmr:
            return self._apply_per_paper_capping(chunks, k, max_per_paper)
        
        # Encode all chunk texts for diversity computation (similarity between chunks)
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings = self.embedding_model.encode(
            chunk_texts,
            convert_to_tensor=True,
            batch_size=settings.embedding_batch_size
        )
        
        # Normalize embeddings for cosine similarity
        chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
        
        # Use existing scores (rerank_score or score) as relevance if available
        # Otherwise compute from embeddings
        use_existing_scores = any(chunk.rerank_score is not None for chunk in chunks) or \
                             any(chunk.score is not None for chunk in chunks)
        
        if use_existing_scores:
            # Use existing scores (rerank_score preferred, fallback to score)
            relevance_scores = [
                chunk.rerank_score if chunk.rerank_score is not None else chunk.score
                for chunk in chunks
            ]
            # Normalize relevance scores to [0, 1] for fair comparison
            max_relevance = max(relevance_scores) if relevance_scores else 1.0
            min_relevance = min(relevance_scores) if relevance_scores else 0.0
            if max_relevance > min_relevance:
                relevance_scores = [
                    (score - min_relevance) / (max_relevance - min_relevance)
                    for score in relevance_scores
                ]
        else:
            # Compute relevance from embeddings
            question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
            question_embedding = F.normalize(question_embedding, p=2, dim=1)
            relevance_scores = torch.mm(question_embedding, chunk_embeddings.t()).squeeze(0)
            relevance_scores = [float(score) for score in relevance_scores]
        
        # Track selected chunks and per-paper counts
        selected = []
        selected_indices = set()
        paper_counts = {}  # paper_id -> count
        
        # Start with the highest relevance chunk
        if chunks:
            # Find the chunk with highest relevance score
            first_idx = 0
            max_relevance = relevance_scores[0] if isinstance(relevance_scores, list) else float(relevance_scores[0])
            for idx in range(1, len(chunks)):
                relevance = relevance_scores[idx] if isinstance(relevance_scores, list) else float(relevance_scores[idx])
                if relevance > max_relevance:
                    max_relevance = relevance
                    first_idx = idx
            
            first_chunk = chunks[first_idx]
            selected.append(first_chunk)
            selected_indices.add(first_idx)
            paper_counts[first_chunk.paper_id] = paper_counts.get(first_chunk.paper_id, 0) + 1
        
        # Select remaining chunks using MMR
        while len(selected) < k and len(selected) < len(chunks):
            best_mmr_score = float('-inf')
            best_idx = None
            
            for idx, chunk in enumerate(chunks):
                # Skip if already selected
                if idx in selected_indices:
                    continue
                
                # Skip if paper limit reached
                paper_count = paper_counts.get(chunk.paper_id, 0)
                if paper_count >= max_per_paper:
                    continue
                
                # Compute relevance component
                relevance = relevance_scores[idx] if isinstance(relevance_scores, list) else float(relevance_scores[idx])
                
                # Compute diversity component (max similarity to already selected chunks)
                max_similarity = 0.0
                if selected:
                    # Compute similarity to all selected chunks
                    selected_embeddings = chunk_embeddings[[i for i in selected_indices]]
                    current_embedding = chunk_embeddings[idx:idx+1]
                    similarities = torch.mm(current_embedding, selected_embeddings.t()).squeeze(0)
                    max_similarity = float(torch.max(similarities))
                
                # MMR score: λ * relevance - (1-λ) * max_similarity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            # If no valid chunk found (all papers at limit), break
            if best_idx is None:
                break
            
            # Add the best chunk
            best_chunk = chunks[best_idx]
            selected.append(best_chunk)
            selected_indices.add(best_idx)
            paper_counts[best_chunk.paper_id] = paper_counts.get(best_chunk.paper_id, 0) + 1
        
        logger.info(f"✓ MMR applied: selected {len(selected)} chunks from {len(paper_counts)} unique papers")
        logger.debug(f"   Per-paper distribution: {dict(sorted(paper_counts.items(), key=lambda x: x[1], reverse=True))}")
        
        return selected
    
    def _apply_per_paper_capping(
        self,
        chunks: List[RetrievedChunk],
        k: int,
        max_per_paper: int
    ) -> List[RetrievedChunk]:
        """
        Apply per-paper chunk capping without MMR.
        
        Args:
            chunks: List of retrieved chunks (should be sorted by relevance score)
            k: Number of chunks to return
            max_per_paper: Maximum chunks per paper
        
        Returns:
            List of chunks with per-paper capping applied
        """
        selected = []
        paper_counts = {}  # paper_id -> count
        
        for chunk in chunks:
            if len(selected) >= k:
                break
            
            paper_count = paper_counts.get(chunk.paper_id, 0)
            if paper_count < max_per_paper:
                selected.append(chunk)
                paper_counts[chunk.paper_id] = paper_count + 1
        
        logger.info(f"✓ Per-paper capping applied: selected {len(selected)} chunks from {len(paper_counts)} unique papers")
        logger.debug(f"   Per-paper distribution: {dict(sorted(paper_counts.items(), key=lambda x: x[1], reverse=True))}")
        
        return selected
    
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
        
        # For MMR, we need more candidates to select from
        # Fetch more candidates if MMR is enabled or per-paper capping is used
        fetch_k = k
        if settings.use_mmr or settings.max_chunks_per_paper < 1000:  # 1000 is effectively no limit
            # Fetch 2-3x more candidates to have enough diversity for MMR selection
            fetch_k = max(k * 3, k + 10)
        
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
            
            # Perform separate searches for dense and sparse vectors using the 'using' parameter
            try:
                # Search dense vectors
                dense_results = self.client.query_points(
                    collection_name=settings.qdrant_collection_name,
                    query=dense_embedding.tolist(),
                    using="dense",
                    query_filter=query_filter,
                    limit=fetch_k,
                    with_payload=True
                ).points
                
                # Search sparse vectors
                sparse_results = self.client.query_points(
                    collection_name=settings.qdrant_collection_name,
                    query=sparse_vector,
                    using="sparse",
                    query_filter=query_filter,
                    limit=fetch_k,
                    with_payload=True
                ).points
            except (TypeError, ValueError, AttributeError) as e:
                # Fallback: If named vector format doesn't work, try without names
                logger.warning(f"⚠ Warning: Named vector search failed ({e}), trying fallback...")
                try:
                    dense_results = self.client.query_points(
                        collection_name=settings.qdrant_collection_name,
                        query=dense_embedding.tolist(),
                        using="dense",
                        query_filter=query_filter,
                        limit=k * 2,
                        with_payload=True
                    ).points
                except Exception:
                    # If even fallback fails, raise the original error
                    raise ValueError(
                        f"Failed to search collection. Collection requires named vectors (dense, sparse). "
                        f"Error: {e}"
                    )
                # For sparse, we can't use regular search, so just use dense
                sparse_results = []
            
            # Fusion method: RRF or weighted score fusion
            from collections import defaultdict
            all_results = {}
            
            # Store results by ID with their scores
            dense_scores = {}
            sparse_scores = {}
            
            for result in dense_results:
                dense_scores[result.id] = result.score
                all_results[result.id] = result
            
            for result in sparse_results:
                sparse_scores[result.id] = result.score
                if result.id not in all_results:
                    all_results[result.id] = result
            
            if settings.hybrid_search_fusion_method == "weighted":
                # Weighted score fusion: α·dense + (1-α)·sparse
                alpha = settings.hybrid_search_alpha
                fused_scores = {}
                
                # Normalize scores to [0, 1] range for fair combination
                # Get max scores for normalization
                dense_max = max(dense_scores.values()) if dense_scores else 1.0
                sparse_max = max(sparse_scores.values()) if sparse_scores else 1.0
                
                # Normalize and combine scores
                all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
                for result_id in all_ids:
                    dense_score = dense_scores.get(result_id, 0.0)
                    sparse_score = sparse_scores.get(result_id, 0.0)
                    
                    # Normalize to [0, 1]
                    norm_dense = dense_score / dense_max if dense_max > 0 else 0.0
                    norm_sparse = sparse_score / sparse_max if sparse_max > 0 else 0.0
                    
                    # Weighted fusion: α·dense + (1-α)·sparse
                    fused_scores[result_id] = alpha * norm_dense + (1 - alpha) * norm_sparse
                
                # Sort by fused score and take top fetch_k
                sorted_by_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
                top_k_ids = [result_id for result_id, _ in sorted_by_fused[:fetch_k]]
                
                # Get results in fused score order
                search_results = [all_results[result_id] for result_id in top_k_ids if result_id in all_results]
                
                # Update scores to fused scores
                for result in search_results:
                    result.score = fused_scores.get(result.id, 0.0)
            else:
                # RRF (Reciprocal Rank Fusion) - original method
                rrf_scores = defaultdict(float)
                
                # Calculate RRF scores from dense results
                for rank, result in enumerate(dense_results, 1):
                    rrf_scores[result.id] += 1.0 / (60 + rank)
                
                # Calculate RRF scores from sparse results
                for rank, result in enumerate(sparse_results, 1):
                    rrf_scores[result.id] += 1.0 / (60 + rank)
                
                # Sort by RRF score and take top fetch_k
                sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                top_k_ids = [result_id for result_id, _ in sorted_by_rrf[:fetch_k]]
                
                # Get results in RRF order
                search_results = [all_results[result_id] for result_id in top_k_ids if result_id in all_results]
                
                # Update scores to RRF scores
                for result in search_results:
                    result.score = rrf_scores.get(result.id, 0.0)
        else:
            # Dense-only search (backward compatible)
            question_embedding = self.embedding_model.encode(question)
            
            search_results = self.client.query_points(
                collection_name=settings.qdrant_collection_name,
                query=question_embedding.tolist(),
                query_filter=query_filter,
                limit=fetch_k,
                with_payload=True
            ).points
        
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
                initial_score=result.score,  # For non-reranked results, initial_score = score
                rerank_score=None,  # No reranking in this method
                modality_tags=payload.get("modality_tags", []),
                authors=payload.get("authors") if payload.get("authors") else []  # Authors if available in payload
            )
            retrieved_chunks.append(chunk)
        
        # Apply MMR and per-paper capping
        if settings.use_mmr or settings.max_chunks_per_paper < 1000:
            retrieved_chunks = self._apply_mmr_and_capping(
                chunks=retrieved_chunks,
                question=question,
                k=k,
                lambda_param=settings.mmr_lambda,
                max_per_paper=settings.max_chunks_per_paper
            )
        
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
        
        # Store initial scores before reranking
        for chunk in chunks:
            chunk.initial_score = chunk.score
        
        # If no reranker is available or not enough chunks, return as-is
        if not self.use_reranking or not self.reranker or len(chunks) <= 1:
            return chunks[:final_k]
        
        # Step 2: Rerank using cross-encoder
        try:
            # Prepare pairs of [question, chunk_text] for the cross-encoder
            pairs = [[question, chunk.text] for chunk in chunks]
            
            # Get reranking scores from cross-encoder
            rerank_scores = self.reranker.predict(pairs)
            
            # Update chunk scores with reranking scores and store both
            for chunk, rerank_score in zip(chunks, rerank_scores):
                chunk.rerank_score = float(rerank_score)
                chunk.score = float(rerank_score)  # Final score is rerank score
            
            # Sort by reranker scores (higher is better)
            chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
            
            # Apply MMR and per-paper capping after reranking
            # Use rerank scores as relevance scores for MMR
            if settings.use_mmr or settings.max_chunks_per_paper < 1000:
                chunks = self._apply_mmr_and_capping(
                    chunks=chunks,
                    question=question,
                    k=final_k,
                    lambda_param=settings.mmr_lambda,
                    max_per_paper=settings.max_chunks_per_paper
                )
            else:
                # Just apply per-paper capping if MMR is disabled
                chunks = chunks[:final_k]
            
            return chunks
        
        except Exception as e:
            logger.warning(f"⚠ Reranking failed: {e}")
            logger.info("  Returning original retrieval results")
            # Apply MMR and per-paper capping even if reranking failed
            if settings.use_mmr or settings.max_chunks_per_paper < 1000:
                chunks = self._apply_mmr_and_capping(
                    chunks=chunks,
                    question=question,
                    k=final_k,
                    lambda_param=settings.mmr_lambda,
                    max_per_paper=settings.max_chunks_per_paper
                )
            else:
                chunks = chunks[:final_k]
            return chunks


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


