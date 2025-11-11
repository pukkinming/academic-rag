"""FastAPI server for the academic RAG system."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import uvicorn
import os

from config import settings
from models import QuestionRequest, AnswerResponse, Source, RetrievedChunk
from retriever import Retriever
from prompt_builder import build_messages, extract_sources_from_chunks
from llm_client import get_llm_client


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


@app.on_event("startup")
async def startup_event():
    """Initialize components on server startup."""
    global retriever, llm_client
    
    print("=" * 80)
    print("🚀 Starting Emotion Recognition RAG API")
    print("=" * 80)
    
    # Initialize retriever
    print("\n📚 Initializing retriever...")
    try:
        retriever = Retriever()
        print(f"✓ Retriever initialized with model: {settings.embedding_model}")
    except Exception as e:
        print(f"✗ Failed to initialize retriever: {e}")
        raise
    
    # Initialize LLM client
    print("\n🤖 Initializing LLM client...")
    try:
        llm_client = get_llm_client()
        print(f"✓ LLM client initialized: {llm_client.get_info()}")
    except Exception as e:
        print(f"✗ Failed to initialize LLM client: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("✓ Server ready!")
    print(f"📍 Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"📍 Collection: {settings.qdrant_collection_name}")
    print(f"🤖 LLM Provider: {settings.llm_provider}")
    print(f"🤖 LLM Model: {settings.llm_model}")
    print("=" * 80 + "\n")


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
    """
    try:
        # Retrieve chunks
        chunks = retriever.retrieve(
            question=request.question,
            k=request.top_k,
            year_min=request.year_min,
            year_max=request.year_max,
            modality_tags=request.modality_tags
        )
        
        # Convert to dict format
        results = []
        for chunk in chunks:
            results.append({
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
            })
        
        return {
            "question": request.question,
            "retrieved_chunks": len(results),
            "chunks": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """
    Ask a question and get an answer with citations.
    
    Flow:
    1. Retrieve relevant chunks from vector database
    2. Build academic prompt with evidence
    3. Call LLM to generate answer
    4. Extract source citations
    5. Return structured response
    """
    try:
        # Step 1: Retrieve relevant chunks
        chunks = retriever.retrieve(
            question=request.question,
            k=request.top_k,
            year_min=request.year_min,
            year_max=request.year_max,
            modality_tags=request.modality_tags
        )
        
        if not chunks:
            # No relevant literature found
            return AnswerResponse(
                answer="This topic is not adequately addressed in the provided literature. No relevant papers were found matching your query and filters.",
                sources=[],
                question=request.question
            )
        
        # Step 2: Build prompt with evidence
        messages = build_messages(request.question, chunks)
        
        # Step 3: Generate answer with LLM
        try:
            answer = llm_client.generate(messages)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"LLM generation failed: {str(e)}"
            )
        
        # Step 4: Extract sources
        sources_data = extract_sources_from_chunks(chunks)
        sources = [Source(**src) for src in sources_data]
        
        # Step 5: Return response
        return AnswerResponse(
            answer=answer,
            sources=sources,
            question=request.question
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/ask-stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a question and get a streaming answer.
    
    Returns a streaming response with server-sent events (SSE).
    """
    try:
        # Step 1: Retrieve relevant chunks
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
