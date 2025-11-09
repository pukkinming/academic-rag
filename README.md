# Academic RAG System

A minimal, controlled RAG (Retrieval-Augmented Generation) system for academic literature review in affective computing and emotion recognition. Built with explicit control over every component: chunking, retrieval, prompt construction, and citation formatting.

## Overview

This is **not** a generic, off-the-shelf RAG. It's a domain-specific system designed for:
- **Academic rigor**: Enforces proper citation style (Author, Year)
- **Transparency**: Full control over chunking, retrieval, and prompting
- **Auditability**: Track which papers contribute to each answer
- **Domain awareness**: Metadata filtering by year, modality tags, etc.
- **Flexible LLM support**: Choose between OpenAI API or self-hosted vLLM

### LLM Options

- **OpenAI API** - GPT-4, GPT-3.5-turbo, or Azure OpenAI
- **vLLM** - Self-hosted models (Mistral, Llama-2, Mixtral, etc.) with OpenAI-compatible API

Switch between them by changing a single config variable. See [LLM_CONFIGURATION.md](LLM_CONFIGURATION.md) for details.

## Architecture

The system consists of 4 main components:

### 1. Ingest Script (`ingest.py`)
- Reads parsed JSON papers
- Chunks sections into ~800-1200 tokens with overlap
- Preserves metadata: `section_name`, `paper_id`, `year`, `citation_pointer`, `pages`
- Embeds chunks using sentence-transformers
- Upserts to Qdrant vector database

### 2. Retriever (`retriever.py`)
- Embeds questions using the same model
- Queries Qdrant for top-k similar chunks
- Supports metadata filtering:
  - Year range (`year_min`, `year_max`)
  - Modality tags (gait, facial, speech, etc.)
  - Specific paper IDs
- Returns scored chunks with full metadata

### 3. Prompt Builder (`prompt_builder.py`)
- Constructs academic-style prompts
- System instruction enforces:
  - Mandatory citations for all claims
  - Evidence-only responses (no hallucinations)
  - Academic tone and terminology
  - Explicit acknowledgment of missing information
- Formats evidence chunks with citation headers
- Supports both single-prompt and chat message formats

### 4. API Endpoint (`api.py`)
- FastAPI server with `/ask` endpoint
- Flow:
  1. Retrieve relevant chunks
  2. Build academic prompt
  3. Call LLM (OpenAI or vLLM)
  4. Return answer with source citations
- Additional endpoints for health checks and statistics
- Supports both OpenAI and vLLM backends

## Installation

### Prerequisites
- **Docker** (recommended) OR Python 3.9+
- **Qdrant** vector database (included in Docker setup)
- **OpenAI API key** OR **vLLM** (for self-hosted LLM with GPU)

### Option 1: Docker Setup (Recommended)

**Fastest way to get started:**

```bash
# 1. Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# 2. Start everything
docker-compose up -d

# 3. Verify
curl http://localhost:8000/health

# 4. Ingest example paper
docker-compose exec api python ingest.py example_paper.json

# 5. Query
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does emotion affect gait patterns?"}'
```

**See [DOCKER.md](DOCKER.md) for complete Docker documentation.**

### Option 2: Local Python Setup

1. **Clone and navigate to the project:**
```bash
cd /home/frank/Dropbox/Project/academic-rag
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Qdrant:**

Option A: Docker
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Option B: Local installation
```bash
# Follow instructions at https://qdrant.tech/documentation/quick-start/
```

4. **Configure environment:**

Create a `.env` file:
```bash
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=emotion_recognition_papers

# Embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM - Option 1: OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
OPENAI_API_KEY=your-key-here

# LLM - Option 2: vLLM (self-hosted)
# LLM_PROVIDER=vllm
# LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
# VLLM_BASE_URL=http://localhost:8000/v1

# Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_TOP_K=8
```

## Usage

### 1. Prepare Paper Data

Your papers should be in JSON format. Example structure:

```json
{
  "paper_id": "narayanan2020",
  "title": "Proxemic Fusion: Emotion Recognition from Gait",
  "authors": ["Narayanan, A.", "Smith, B.", "Johnson, C."],
  "year": 2020,
  "citation_pointer": "Narayanan et al., 2020",
  "venue": "IEEE TAFFC",
  "modality_tags": ["gait", "emotion_recognition"],
  "sections": [
    {
      "section_name": "I. Introduction",
      "text": "Emotion recognition from gait patterns has gained...",
      "page_start": 1,
      "page_end": 2
    },
    {
      "section_name": "II. Related Work",
      "text": "Previous work on affective computing...",
      "page_start": 2,
      "page_end": 3
    }
  ]
}
```

Multiple papers can be in a JSON array:
```json
[
  { "paper_id": "paper1", ... },
  { "paper_id": "paper2", ... }
]
```

### 2. Ingest Papers

```bash
python ingest.py path/to/papers.json
```

This will:
- Chunk each section
- Generate embeddings
- Store in Qdrant with metadata

Output example:
```
Loading papers from: papers.json
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Connecting to Qdrant at localhost:6333

Ingesting paper: Narayanan et al., 2020 - Proxemic Fusion...
  Created 15 chunks
  Embedding chunks...
  Upserting to Qdrant...
  ✓ Successfully ingested 15 chunks

============================================================
Ingestion complete!
  Papers ingested: 5
  Total chunks: 73
  Avg chunks per paper: 14.6
============================================================
```

### 3. Start the API Server

```bash
python api.py
```

Or with uvicorn:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Query the System

#### Using curl:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does emotion affect gait patterns in crowded environments?",
    "top_k": 8,
    "year_min": 2018
  }'
```

#### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "Explain proxemic fusion and emotion classification",
        "top_k": 10,
        "modality_tags": ["gait"]
    }
)

result = response.json()
print("Answer:", result["answer"])
print("\nSources:")
for src in result["sources"]:
    print(f"  - {src['citation']}: {src['title']}")
```

#### Response format:

```json
{
  "answer": "Proxemic fusion is a framework that combines spatial proximity...(Narayanan et al., 2020). The approach improves emotion classification by...(Randhavane et al., 2019).",
  "sources": [
    {
      "paper_id": "narayanan2020",
      "citation": "Narayanan et al., 2020",
      "section": "I. Introduction",
      "pages": [1, 2],
      "title": "Proxemic Fusion: Emotion Recognition from Gait"
    }
  ],
  "question": "Explain proxemic fusion and emotion classification"
}
```

## API Endpoints

### `POST /ask`
Ask questions about the literature.

**Request body:**
```json
{
  "question": "string (required)",
  "top_k": "integer (default: 8)",
  "year_min": "integer (optional)",
  "year_max": "integer (optional)",
  "modality_tags": ["string"] (optional)
}
```

### `POST /retrieve`
Retrieve chunks without generating an answer (for debugging).

### `GET /health`
Health check - verify Qdrant connection and collection status.

### `GET /stats`
Get collection statistics (total chunks, vector size, etc.).

### `GET /`
API information and available endpoints.

## Configuration

All settings are in `config.py` and can be overridden via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server host | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `QDRANT_COLLECTION_NAME` | Collection name | `emotion_recognition_papers` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `LLM_PROVIDER` | LLM provider (openai/vllm) | `openai` |
| `LLM_MODEL` | Model name | `gpt-4-turbo-preview` |
| `CHUNK_SIZE` | Target chunk size (tokens) | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks (tokens) | `200` |
| `DEFAULT_TOP_K` | Default retrieval count | `8` |

## Testing the Retriever

You can test retrieval without the full API:

```python
from retriever import Retriever

retriever = Retriever()

# Basic retrieval
chunks = retriever.retrieve(
    question="How does emotion affect navigation?",
    k=5
)

# With filters
chunks = retriever.retrieve(
    question="Facial expression recognition methods",
    k=10,
    year_min=2020,
    modality_tags=["facial"]
)

# Print results
for i, chunk in enumerate(chunks, 1):
    print(f"\n{i}. [{chunk.citation_pointer}] (score: {chunk.score:.4f})")
    print(f"   Section: {chunk.section_name}")
    print(f"   {chunk.text[:200]}...")
```

## Testing the Prompt Builder

```python
from prompt_builder import build_prompt, build_messages
from retriever import Retriever

retriever = Retriever()
chunks = retriever.retrieve("How does emotion affect gait?", k=5)

# Single-string prompt (for completion models)
prompt = build_prompt("How does emotion affect gait?", chunks)
print(prompt)

# Chat messages format (for chat models)
messages = build_messages("How does emotion affect gait?", chunks)
for msg in messages:
    print(f"[{msg['role']}]: {msg['content']}")
```

## Project Structure

```
academic-rag/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration management
├── models.py                 # Pydantic data models
├── ingest.py                 # Ingestion script
├── retriever.py              # Retrieval logic
├── prompt_builder.py         # Prompt construction
├── api.py                    # FastAPI server
├── data/                     # Your paper JSON files (create this)
│   ├── papers.json
│   └── ...
└── .env                      # Environment variables (create this)
```

## Why This Approach?

### What You Get:
✅ **Control**: Every decision is explicit and tunable  
✅ **Citations**: Enforced academic citation style  
✅ **Filtering**: Domain-aware metadata filtering  
✅ **Auditability**: Know exactly which papers contributed  
✅ **No black boxes**: Understand and debug every step  

### What You Avoid:
❌ Generic prompts that don't enforce citations  
❌ Chunking strategies that lose context  
❌ Hallucinations from mixing evidence with world knowledge  
❌ Opaque scoring that can't be explained  

## Next Steps

Once you have the baseline working:

1. **Tune chunking**: Experiment with chunk size and overlap
2. **Add reranking**: Use cross-encoders for second-stage ranking
3. **Hybrid search**: Combine dense embeddings with sparse (BM25) retrieval
4. **Fine-tune embeddings**: Domain-specific embedding models
5. **AutoRAG**: Automated optimization of retrieval parameters
6. **MCP integration**: Expose as Model Context Protocol tool
7. **Evaluation**: Build a test set and measure citation accuracy

## Troubleshooting

### "Collection not found"
Run the ingest script first to create the collection.

### "No relevant literature found"
- Check that papers have been ingested
- Try a broader question
- Remove or relax filters

### "Error calling LLM"
- Verify API key (OpenAI) or vLLM server is running
- Check `LLM_PROVIDER` and `LLM_MODEL` settings
- For vLLM: verify VLLM_BASE_URL is correct and server is accessible

### Embeddings are slow
- Use a smaller embedding model
- Batch process during ingestion
- Consider GPU acceleration

### Citations are not showing up
- Verify the prompt builder is being used
- Check that `citation_pointer` is set in paper JSON
- Review the LLM temperature (lower = more adherent)

## License

MIT License - Use freely for academic research.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{emotion_rag_2025,
  title={Emotion Recognition RAG System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/academic-rag}
}
```

---

**Built for researchers who need control, not convenience.**

