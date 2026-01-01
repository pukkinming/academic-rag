"""FastAPI server for the academic RAG system."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import uvicorn
import os
import logging
from datetime import datetime

from config import settings
from models import QuestionRequest, AnswerResponse, Source, RetrievedChunk
from retriever import Retriever
from prompt_builder import build_messages, extract_sources_from_chunks
from llm_client import get_llm_client
from evaluation import AnswerQualityEvaluator, AnswerValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('api.log', mode='a')  # File output
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy watchfiles logs (from uvicorn auto-reload)
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)


# Initialize FastAPI app
app = FastAPI(
    title="Academic RAG API",
    description="Retrieval-Augmented Generation system for academic literature",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global instances (initialized on startup)
retriever = None
llm_client = None
quality_evaluator = None
answer_validator = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on server startup."""
    global retriever, llm_client, quality_evaluator, answer_validator
    
    logger.info("=" * 80)
    logger.info("🚀 Starting Emotion Recognition RAG API")
    logger.info("=" * 80)
    
    # Initialize retriever
    logger.info("📚 Initializing retriever...")
    try:
        retriever = Retriever()
        logger.info(f"✓ Retriever initialized with model: {settings.embedding_model}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize retriever: {e}", exc_info=True)
        raise
    
    # Initialize LLM client
    logger.info("🤖 Initializing LLM client...")
    try:
        llm_client = get_llm_client()
        logger.info(f"✓ LLM client initialized: {llm_client.get_info()}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize LLM client: {e}", exc_info=True)
        raise
    
    # Initialize quality evaluation components
    logger.info("📊 Initializing quality evaluators...")
    try:
        # Share embedding model with retriever for efficiency
        quality_evaluator = AnswerQualityEvaluator(embedding_model=retriever.embedding_model)
        answer_validator = AnswerValidator(embedding_model=retriever.embedding_model)
        logger.info("✓ Quality evaluators initialized")
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize quality evaluators: {e}")
        logger.info("  /ask-with-evaluation endpoint will use lazy initialization")
    
    logger.info("=" * 80)
    logger.info("✓ Server ready!")
    logger.info(f"📍 Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"📍 Collection: {settings.qdrant_collection_name}")
    logger.info(f"🤖 LLM Provider: {settings.llm_provider}")
    logger.info(f"🤖 LLM Model: {settings.llm_model}")
    logger.info("=" * 80)


@app.get("/")
async def root():
    """Serve the web UI or return API information."""
    static_index = os.path.join(static_dir, "index.html")
    if os.path.exists(static_index):
        return FileResponse(static_index)
    else:
        # Fallback to JSON if static files don't exist
        return {
            "name": "Academic RAG API",
            "version": "1.0.0",
            "description": "Retrieval-Augmented Generation system for academic literature",
            "endpoints": {
                "POST /ask": "Ask a question with RAG",
                "POST /retrieve": "Retrieve relevant chunks (no LLM)",
                "GET /health": "Health check",
                "GET /stats": "Collection statistics",
                "GET /llm-info": "LLM configuration info"
            },
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model
        }


@app.get("/api")
async def api_info() -> Dict[str, Any]:
    """API information endpoint."""
    return {
        "name": "Academic RAG API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation system for academic literature",
        "endpoints": {
            "POST /ask": "Ask a question with RAG",
            "POST /retrieve": "Retrieve relevant chunks (no LLM)",
            "GET /health": "Health check",
            "GET /stats": "Collection statistics",
            "GET /llm-info": "LLM configuration info"
        },
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Verifies:
    - Qdrant connection
    - Collection exists
    - LLM client initialized
    """
    try:
        # Check Qdrant connection
        collections = retriever.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        collection_exists = settings.qdrant_collection_name in collection_names
        
        # Get basic collection info if it exists
        # Note: Using count() instead of get_collection() to avoid version mismatch issues
        collection_info = None
        if collection_exists:
            try:
                count_result = retriever.client.count(
                    collection_name=settings.qdrant_collection_name
                )
                collection_info = {
                    "name": settings.qdrant_collection_name,
                    "points_count": count_result.count if count_result else 0
                }
            except Exception as e:
                # Fallback to basic info if count fails
                collection_info = {
                    "name": settings.qdrant_collection_name,
                    "exists": True
                }
        
        # Check LLM client
        llm_info = llm_client.get_info() if llm_client else None
        
        return {
            "status": "healthy" if collection_exists else "collection_not_found",
            "qdrant": {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "connected": True,
                "collections": collection_names
            },
            "collection": collection_info,
            "llm": llm_info,
            "ready": collection_exists and llm_client is not None
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get collection statistics."""
    try:
        # Use count() instead of get_collection() to avoid version mismatch issues
        count_result = retriever.client.count(
            collection_name=settings.qdrant_collection_name
        )
        
        return {
            "collection_name": settings.qdrant_collection_name,
            "points_count": count_result.count if count_result else 0,
            "config": {
                "embedding_model": settings.embedding_model,
                "qdrant_host": settings.qdrant_host,
                "qdrant_port": settings.qdrant_port
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")


@app.get("/llm-info")
async def get_llm_info() -> Dict[str, Any]:
    """Get LLM configuration information."""
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    return llm_client.get_info()


@app.post("/retrieve")
async def retrieve_chunks(request: QuestionRequest) -> Dict[str, Any]:
    """
    Retrieve relevant chunks without generating an answer.
    
    Useful for:
    - Debugging retrieval quality
    - Understanding which papers are being retrieved
    - Inspecting relevance scores
    
    Supports optional reranking for improved quality.
    """
    logger.info(f"🔍 /retrieve request: {request.question[:100]}...")
    
    try:
        # Determine if reranking should be used
        use_reranking = request.use_reranking if request.use_reranking is not None else settings.use_reranking
        logger.info(f"🔄 Reranking: {'enabled' if use_reranking else 'disabled'}")
        
        # Retrieve chunks (with or without reranking)
        if use_reranking and retriever.use_reranking:
            logger.info(f"   Retrieving with reranking (initial_k={request.rerank_initial_k or settings.rerank_initial_k})")
            chunks = retriever.retrieve_with_reranking(
                question=request.question,
                k=request.top_k,
                initial_k=request.rerank_initial_k,
                year_min=request.year_min,
                year_max=request.year_max,
                modality_tags=request.modality_tags
            )
        else:
            logger.info(f"   Retrieving without reranking (top_k={request.top_k})")
            chunks = retriever.retrieve(
                question=request.question,
                k=request.top_k,
                year_min=request.year_min,
                year_max=request.year_max,
                modality_tags=request.modality_tags
            )
        
        logger.info(f"✓ Retrieved {len(chunks)} chunks")
        # logger.info(f"   Chunks: {chunks}")
        
        # Convert to dict format
        results = []
        for chunk in chunks:
            result = {
                "text": chunk.text,
                "title": chunk.title,
                "citation_pointer": chunk.citation_pointer,
                "section_name": chunk.section_name,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "paper_id": chunk.paper_id,
                "year": chunk.year,
                "score": chunk.score,
                "modality_tags": chunk.modality_tags,
                "authors": chunk.authors
            }
            # Add reranking information if available
            if chunk.initial_score is not None:
                result["initial_score"] = chunk.initial_score
            if chunk.rerank_score is not None:
                result["rerank_score"] = chunk.rerank_score
            
            results.append(result)
        
        return {
            "question": request.question,
            "retrieved_chunks": len(results),
            "chunks": results,
            "reranking_used": use_reranking and retriever.use_reranking
        }
    
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """
    Ask a question and get an answer with citations.
    
    Flow:
    1. Retrieve relevant chunks from vector database (with optional reranking)
    2. Build academic prompt with evidence
    3. Call LLM to generate answer
    4. Extract source citations
    5. Return structured response
    
    Supports optional reranking for improved retrieval quality.
    """
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info(f"📝 New /ask request received")
    logger.info(f"Question: {request.question}")
    logger.info(f"Parameters: top_k={request.top_k}, year_min={request.year_min}, "
                f"year_max={request.year_max}, modality_tags={request.modality_tags}")
    
    try:
        # Determine if reranking should be used
        use_reranking = request.use_reranking if request.use_reranking is not None else settings.use_reranking
        logger.info(f"🔄 Reranking: {'enabled' if use_reranking else 'disabled'}")
        
        # Step 1: Retrieve relevant chunks (with or without reranking)
        logger.info("🔍 Step 1: Starting retrieval...")
        retrieval_start = datetime.now()
        
        if use_reranking and retriever.use_reranking:
            initial_k = request.rerank_initial_k or settings.rerank_initial_k
            logger.info(f"   Using reranking: fetching {initial_k} candidates → reranking to top {request.top_k}")
            chunks = retriever.retrieve_with_reranking(
                question=request.question,
                k=request.top_k,
                initial_k=request.rerank_initial_k,
                year_min=request.year_min,
                year_max=request.year_max,
                modality_tags=request.modality_tags
            )
        else:
            logger.info(f"   Using standard retrieval: fetching top {request.top_k}")
            chunks = retriever.retrieve(
                question=request.question,
                k=request.top_k,
                year_min=request.year_min,
                year_max=request.year_max,
                modality_tags=request.modality_tags
            )
        
        retrieval_time = (datetime.now() - retrieval_start).total_seconds()
        logger.info(f"✓ Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        logger.debug(f"   Chunks: {chunks}")
        
        if chunks:
            logger.debug(f"   Top chunk scores: {[f'{c.score:.4f}' for c in chunks[:3]]}")
            unique_papers = len(set(c.paper_id for c in chunks))
            logger.info(f"   Chunks from {unique_papers} unique papers")
        
        if not chunks:
            logger.warning("⚠️  No relevant chunks found for query")
            logger.info("=" * 80)
            return AnswerResponse(
                answer="This topic is not adequately addressed in the provided literature. No relevant papers were found matching your query and filters.",
                sources=[],
                question=request.question
            )
        
        # Step 2: Build prompt with evidence
        logger.info("📋 Step 2: Building prompt with evidence...")
        messages = build_messages(request.question, chunks)
        prompt_tokens = sum(len(str(m)) for m in messages) // 4  # Rough estimate
        logger.info(f"   Prompt size: ~{prompt_tokens} tokens")
        logger.debug(f"   Messages: {messages}")
        
        # Step 3: Generate answer with LLM
        logger.info("🤖 Step 3: Generating answer with LLM...")
        llm_start = datetime.now()
        try:
            answer = llm_client.generate(messages)
            llm_time = (datetime.now() - llm_start).total_seconds()
            logger.info(f"✓ LLM generation completed in {llm_time:.2f}s")
            logger.info(f"   Answer length: {len(answer)} characters")
            logger.debug(f"   Answer: {answer}")
        except Exception as e:
            logger.error(f"✗ LLM generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"LLM generation failed: {str(e)}"
            )
        
        # Step 4: Extract sources
        logger.info("📚 Step 4: Extracting source citations...")
        sources_data = extract_sources_from_chunks(chunks)
        sources = [Source(**src) for src in sources_data]
        logger.info(f"✓ Extracted {len(sources)} unique sources")
        logger.debug(f"   Sources: {sources}")
        
        # Step 5: Prepare chunk details with reranking information (only if requested)
        chunk_details = None
        if request.include_chunks:
            chunk_details = []
            for chunk in chunks:
                chunk_dict = {
                    "text": chunk.text,
                    "title": chunk.title,
                    "citation_pointer": chunk.citation_pointer,
                    "section_name": chunk.section_name,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "paper_id": chunk.paper_id,
                    "year": chunk.year,
                    "score": chunk.score,
                    "modality_tags": chunk.modality_tags or [],
                    "authors": chunk.authors if chunk.authors else []
                }
                # Add reranking information if available
                if chunk.initial_score is not None:
                    chunk_dict["initial_score"] = chunk.initial_score
                if chunk.rerank_score is not None:
                    chunk_dict["rerank_score"] = chunk.rerank_score
                
                chunk_details.append(chunk_dict)
        
        # Step 6: Return response
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Request completed successfully in {total_time:.2f}s")
        logger.info(f"   Breakdown: Retrieval={retrieval_time:.2f}s, LLM={llm_time:.2f}s")
        logger.info("=" * 80)
        
        return AnswerResponse(
            answer=answer,
            sources=sources,
            question=request.question,
            chunks=chunk_details
        )
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Request failed after {total_time:.2f}s: {e}", exc_info=True)
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/ask-with-evaluation")
async def ask_question_with_evaluation(request: QuestionRequest) -> Dict[str, Any]:
    """
    Ask a question and get an answer with quality evaluation metrics.
    
    This endpoint extends /ask with:
    - Quality metrics: faithfulness, coverage, utility scores
    - Validation results: citation verification, hallucination detection
    - Confidence scoring
    
    Use this endpoint when you need to assess answer quality programmatically.
    
    Flow:
    1. Retrieve relevant chunks from vector database (with optional reranking)
    2. Build academic prompt with evidence
    3. Call LLM to generate answer
    4. Extract source citations
    5. Evaluate answer quality (faithfulness, coverage, utility)
    6. Validate answer (citations, hallucinations)
    7. Return structured response with metrics
    """
    global quality_evaluator, answer_validator
    
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info(f"📝 New /ask-with-evaluation request received")
    logger.info(f"Question: {request.question}")
    logger.info(f"Parameters: top_k={request.top_k}, year_min={request.year_min}, "
                f"year_max={request.year_max}, modality_tags={request.modality_tags}")
    
    try:
        # Determine if reranking should be used
        use_reranking = request.use_reranking if request.use_reranking is not None else settings.use_reranking
        logger.info(f"🔄 Reranking: {'enabled' if use_reranking else 'disabled'}")
        
        # Step 1: Retrieve relevant chunks (with or without reranking)
        logger.info("🔍 Step 1: Starting retrieval...")
        retrieval_start = datetime.now()
        
        try:
            if use_reranking and retriever.use_reranking:
                initial_k = request.rerank_initial_k or settings.rerank_initial_k
                logger.info(f"   Using reranking: fetching {initial_k} candidates → reranking to top {request.top_k}")
                chunks = retriever.retrieve_with_reranking(
                    question=request.question,
                    k=request.top_k,
                    initial_k=request.rerank_initial_k,
                    year_min=request.year_min,
                    year_max=request.year_max,
                    modality_tags=request.modality_tags
                )
            else:
                logger.info(f"   Using standard retrieval: fetching top {request.top_k}")
                chunks = retriever.retrieve(
                    question=request.question,
                    k=request.top_k,
                    year_min=request.year_min,
                    year_max=request.year_max,
                    modality_tags=request.modality_tags
                )
            
            retrieval_time = (datetime.now() - retrieval_start).total_seconds()
            logger.info(f"✓ Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        except Exception as e:
            retrieval_time = (datetime.now() - retrieval_start).total_seconds()
            logger.error(f"✗ Retrieval failed after {retrieval_time:.2f}s: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
        
        if chunks:
            unique_papers = len(set(c.paper_id for c in chunks))
            logger.info(f"   Chunks from {unique_papers} unique papers")
        
        if not chunks:
            logger.warning("⚠️  No relevant chunks found for query")
            logger.info("=" * 80)
            return {
                "answer": "This topic is not adequately addressed in the provided literature. No relevant papers were found matching your query and filters.",
                "sources": [],
                "question": request.question,
                "quality_metrics": {
                    "faithfulness": 0.0,
                    "coverage": 0.0,
                    "utility": 0.0,
                    "hallucination_count": 0,
                },
                "validation": {
                    "is_valid": False,
                    "confidence": 0.0,
                    "citation_errors": [],
                    "missing_citations_count": 0,
                    "hallucination_warnings": []
                }
            }
        
        # Step 2: Build prompt with evidence
        logger.info("📋 Step 2: Building prompt with evidence...")
        messages = build_messages(request.question, chunks)
        
        # Step 3: Generate answer with LLM
        logger.info("🤖 Step 3: Generating answer with LLM...")
        llm_start = datetime.now()
        try:
            answer = llm_client.generate(messages)
            llm_time = (datetime.now() - llm_start).total_seconds()
            logger.info(f"✓ LLM generation completed in {llm_time:.2f}s")
            logger.info(f"   Answer length: {len(answer)} characters")
        except Exception as e:
            logger.error(f"✗ LLM generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"LLM generation failed: {str(e)}"
            )
        
        # Step 4: Extract sources
        logger.info("📚 Step 4: Extracting source citations...")
        sources_data = extract_sources_from_chunks(chunks)
        sources = [Source(**src) for src in sources_data]
        logger.info(f"✓ Extracted {len(sources)} unique sources")
        
        # Step 5: Evaluate answer quality
        logger.info("📊 Step 5: Evaluating answer quality...")
        eval_start = datetime.now()
        
        quality_metrics = None
        validation_result = None
        
        # Initialize evaluators if not already done
        try:
            logger.info("   Checking evaluator initialization...")
            if quality_evaluator is None:
                logger.info("   Creating AnswerQualityEvaluator (lazy init)...")
                quality_evaluator = AnswerQualityEvaluator(embedding_model=retriever.embedding_model)
                logger.info("   ✓ AnswerQualityEvaluator created")
            else:
                logger.info("   ✓ AnswerQualityEvaluator already initialized")
                
            if answer_validator is None:
                logger.info("   Creating AnswerValidator (lazy init)...")
                answer_validator = AnswerValidator(embedding_model=retriever.embedding_model)
                logger.info("   ✓ AnswerValidator created")
            else:
                logger.info("   ✓ AnswerValidator already initialized")
            
            logger.info(f"   Running quality evaluation (answer length: {len(answer)}, chunks: {len(chunks)})...")
            quality_metrics = quality_evaluator.evaluate_comprehensive(
                answer=answer,
                question=request.question,
                chunks=chunks
            )
            coverage_str = f"{quality_metrics.coverage_score:.3f}" if quality_metrics.coverage_score is not None else "N/A"
            logger.info(f"✓ Quality metrics: faithfulness={quality_metrics.faithfulness_score:.3f}, "
                       f"coverage={coverage_str}, utility={quality_metrics.utility_score:.3f}")
        except Exception as e:
            logger.error(f"✗ Quality evaluation failed: {e}", exc_info=True)
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            quality_metrics = None
        
        # Step 6: Validate answer
        logger.info("✅ Step 6: Validating answer...")
        try:
            validation_result = answer_validator.validate_comprehensive(answer, chunks)
            logger.info(f"✓ Validation: valid={validation_result.is_valid}, "
                       f"confidence={validation_result.confidence_score:.3f}")
        except Exception as e:
            logger.error(f"✗ Validation failed: {e}", exc_info=True)
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            validation_result = None
        
        eval_time = (datetime.now() - eval_start).total_seconds()
        logger.info(f"   Evaluation completed in {eval_time:.2f}s")
        
        # Step 7: Prepare chunk details (only if requested)
        chunk_details = None
        if request.include_chunks:
            chunk_details = []
            for chunk in chunks:
                chunk_dict = {
                    "text": chunk.text,
                    "title": chunk.title,
                    "citation_pointer": chunk.citation_pointer,
                    "section_name": chunk.section_name,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "paper_id": chunk.paper_id,
                    "year": chunk.year,
                    "score": chunk.score,
                    "modality_tags": chunk.modality_tags or [],
                    "authors": chunk.authors if chunk.authors else []
                }
                if chunk.initial_score is not None:
                    chunk_dict["initial_score"] = chunk.initial_score
                if chunk.rerank_score is not None:
                    chunk_dict["rerank_score"] = chunk.rerank_score
                chunk_details.append(chunk_dict)
        
        # Step 8: Return response with evaluation
        eval_time = (datetime.now() - eval_start).total_seconds() if 'eval_start' in locals() else 0.0
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Request completed successfully in {total_time:.2f}s")
        logger.info(f"   Breakdown: Retrieval={retrieval_time:.2f}s, LLM={llm_time:.2f}s, Eval={eval_time:.2f}s")
        logger.info("=" * 80)
        
        try:
            response = {
                "answer": answer,
                "sources": [src.model_dump() for src in sources],
                "question": request.question,
                "quality_metrics": quality_metrics.to_dict() if quality_metrics else None,
                "validation": validation_result.to_dict() if validation_result else None,
            }
            
            if chunk_details:
                response["chunks"] = chunk_details
            
            logger.info("   Preparing response...")
            return response
        except Exception as e:
            logger.error(f"✗ Failed to build response: {e}", exc_info=True)
            # Return minimal response even if serialization fails
            return {
                "answer": answer,
                "sources": [],
                "question": request.question,
                "quality_metrics": None,
                "validation": None,
                "error": f"Failed to serialize response: {str(e)}"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Request failed after {total_time:.2f}s: {e}", exc_info=True)
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/ask-stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a question and get a streaming answer.
    
    Returns a streaming response with server-sent events (SSE).
    Supports optional reranking for improved retrieval quality.
    """
    try:
        # Determine if reranking should be used
        use_reranking = request.use_reranking if request.use_reranking is not None else settings.use_reranking
        
        # Step 1: Retrieve relevant chunks (with or without reranking)
        if use_reranking and retriever.use_reranking:
            chunks = retriever.retrieve_with_reranking(
                question=request.question,
                k=request.top_k,
                initial_k=request.rerank_initial_k,
                year_min=request.year_min,
                year_max=request.year_max,
                modality_tags=request.modality_tags
            )
        else:
            chunks = retriever.retrieve(
                question=request.question,
                k=request.top_k,
                year_min=request.year_min,
                year_max=request.year_max,
                modality_tags=request.modality_tags
            )
        
        if not chunks:
            # No relevant literature found
            async def no_results_generator():
                yield "data: This topic is not adequately addressed in the provided literature.\n\n"
            
            return StreamingResponse(
                no_results_generator(),
                media_type="text/event-stream"
            )
        
        # Step 2: Build prompt with evidence
        messages = build_messages(request.question, chunks)
        
        # Step 3: Stream answer
        def stream_generator():
            try:
                for chunk_text in llm_client.generate_streaming(messages):
                    yield f"data: {chunk_text}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


if __name__ == "__main__":
    # Run with uvicorn
    print(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
