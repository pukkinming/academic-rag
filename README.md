# Academic RAG System

![UI appearence](./static/ui_demo.gif)

A minimal, controlled RAG (Retrieval-Augmented Generation) system for academic literature review. Built with explicit control over every component: chunking, retrieval, prompt construction, and citation formatting.

## Overview

This is **not** a generic, off-the-shelf RAG. It's a domain-specific system designed for:
- **Academic rigor**: Enforces proper citation style (Author, Year)
- **Transparency**: Full control over chunking, retrieval, and prompting
- **Auditability**: Track which papers contribute to each answer
- **Domain awareness**: Metadata filtering by year, modality tags, etc.
- **Flexible LLM support**: Choose between OpenAI API or self-hosted vLLM
- **Web UI**: Modern interface for querying without curl commands

**LLM Options:**
- **OpenAI API** - GPT-4, GPT-3.5-turbo, or Azure OpenAI
- **vLLM** - Self-hosted models (Mistral, Llama-2, Mixtral, etc.) with OpenAI-compatible API

Switch between them by changing a single config variable. See [LLM_CONFIGURATION.md](LLM_CONFIGURATION.md) for details.

## Quick Start

### Step 1: Configure Environment

```bash
cp env.example .env
# Edit .env and set OPENAI_API_KEY (or configure LLM_PROVIDER=vllm)
```

### Step 2: Add Papers

**Using Docker:**
```bash
docker-compose exec api python ingest.py ./extracted_data/
```

**Without Docker:**
```bash
pip install -r requirements.txt
python ingest.py ./extracted_data/
```

The ingest script chunks sections (~800-1200 tokens), generates embeddings, and stores in Qdrant with metadata.

### Step 3: Start the System

```bash
bash start.bash
# Or: docker-compose up -d
```

### Step 4: Access the Web UI

- **RAG Interface**: http://localhost:8000/
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Roadmap

### 1) Retrieval Quality (Immediately)

**Hybrid Search Enhancement:**
- Add BM25 (Elasticsearch/Tantivy) alongside Qdrant
- Fuse scores: `α·dense + (1-α)·sparse`
- Currently using Qdrant's built-in BM25; consider external BM25 for more control

**Reranking:** ✅ **IMPLEMENTED**
- Two-stage retrieval: fetch candidates, then rerank with cross-encoder
- Models: `cross-encoder/ms-marco-MiniLM-L-6-v2` (default), with multiple options available
- Improves retrieval quality by 10-30% (MRR/NDCG metrics)
- Configurable via `config.py` or per-request API parameters
- See [RERANKING_GUIDE.md](others/RERANKING_GUIDE.md) for complete documentation

**Diversification:**
- Implement Maximal Marginal Relevance (MMR) to avoid near-duplicate chunks
- Cap per-paper: 2-3 chunks maximum
- Ensures diverse perspectives in answers

**Enhanced Filtering:**
- Respect `published_year`, `modality_tags`, `section_name`
- Filter by section type (e.g., "method", "results", "limitations")
- Multi-level filtering with AND/OR logic

### 2) Context Building

**Section-Aware Chunking:**
- Prefer whole subsections (~800-1200 tokens)
- Carry `section_name` + `pages` metadata
- Maintain section boundaries when possible

**Citation Pointers:**
- Every chunk includes `(Author, Year)` in header
- Keep citation visible in prompt formatting
- Already implemented; verify consistency

**Budgeting Strategy:**
- Pick ~8-12 final chunks after rerank/MMR
- Distribution: 30-40% "methods/results", 20% "datasets", 20% "limitations", remainder "intro/related"
- Ensure balanced coverage of paper sections

### 3) From Retrieval → Reasoning (Synthesis, Not Stitching)

**Add a Tiny Controller:**
```
Plan → Retrieve per subtopic → Synthesize
```

**Implementation:**
1. Ask LLM: "List 3-5 subtopics required to answer X"
2. For each subtopic, run retrieval (with subtopic-specific queries)
3. Generate one mini-synthesis per subtopic (150-250 words, with citations)
4. Generate a final integrated summary (400-700 words) that references those sub-summaries

This 3-step loop is the simplest "agent" and produces literate, structured outputs.

### 4) Prompt Templates That Force Scholarship

**A. Comparative Methods Template:**
```
You are an expert reviewer. Using ONLY the evidence below, write a scholarly comparison.

Required structure:
1) Problem framing (2-3 sentences)
2) Methods compared: bullets with name, core idea, data used, key numbers
3) Strengths vs limitations (one short paragraph)
4) When to use which (decision guidance)

Rules:
- Every factual claim must cite (Author, Year).
- If evidence is missing, say "not reported in the retrieved papers."

[EVIDENCE BLOCKS…]

Question: …

Answer:
```

**B. Dataset Table + Narrative Template:**
```
Task: Build a table of datasets mentioned in evidence.

Columns: Name | Modality | #Subjects | Labels | Setting (lab/in-the-wild) | Known limits

Then write a 150-200 word narrative summary.

Cite (Author, Year) per row or per claim. Use ONLY provided evidence.

[EVIDENCE BLOCKS…]
```

### 5) Faithfulness Guardrails

**Cite-or-Silence:**
- "If not in evidence, say it's unknown"
- Enforce in system prompt (already implemented)

**Attribution Formatting:**
- Include the `citation_pointer` while generating
- Post-process to attach full bibliography later

**Source Caps:**
- Require at least 2 unique papers cited per paragraph when possible
- Prevent over-reliance on single sources

### 6) Evaluation Loop (Don't Skip)

**Create Gold Standard:**
- ~30 gold questions from your domain
- Score on three dimensions:

**Faithfulness (0/1):** Are all claims supported by retrieved text?

**Coverage (0-2):** Did it hit the obvious key works or angles?

**Utility (0-2):** Can this drop into a survey with light edits?

**Automation:**
- Use RAGAS/TruLens or simple rubric + LLM-as-judge
- Manually spot-check 5-10 outputs per change
- Track metrics: reranker swap, chunk size change, etc.

### 7) Auto-Tune Once Baseline is Solid

**After steps 1-6 feel good, run an auto-tuner:**
- AutoRAG-style search on:
  - Chunk size/overlap
  - Embedding model
  - Reranker choice
  - k/MMR parameters
- Use eval set as the objective
- Port best settings back to minimal stack

### 8) Light Instruction-Tuning (Optional but High-Leverage)

**You don't need to "pretrain on PDFs":**
- Do SFT/LoRA to teach tone + structure
- Inputs: (system prompt + evidence + question)
- Outputs: your best, hand-edited answers (with citations)
- 300-1,000 examples are enough for style/discipline on a 7-8B model

**Result:** The model stops being chatty and starts writing like a reviewer.

### 9) Reference Hygiene & Graphs (Nice to Have)

**Store Full References:**
- Full references + a `[n]` → entry map per paper
- When you see `[17]` in evidence, inline-resolve to `(Zhang et al., 2021)`
- Improves readability

**Citation Graph:**
- Build a citation graph to answer "landmark works since 2018 on gait-emotion"
- Visualize paper relationships
- Identify key papers and clusters

### 10) Productize

**API:**
- Keep `/ask` (returns answer, sources[] with paper_id, citation, section, pages)
- Add versioning: `/v1/ask`, `/v2/ask`
- Rate limiting and authentication

**MCP Adapter:**
- Once happy, expose the same `/ask` as an MCP tool
- So Ollama/agents can call it
- Standardize interface for agent frameworks

**Observability:**
- Log: question, retrieved chunk IDs, final used chunks, model, latency
- Track evaluation scores
- Dashboard for monitoring quality over time
- Alert on degradation

### 11) Others
- Dynamically evaluate top_k value from the prompt (rule-based, LLM-based, hybrid)
- add memory
- chunking parameter optimization
- LLM/reranking/instruct model evaluation
- handling bad data/documents in RAG database

## Architecture

The system consists of 4 main components:

1. **Ingest Script** (`ingest.py`) - Chunks papers, generates embeddings (dense + sparse), stores in Qdrant
2. **Retriever** (`retriever.py`) - Hybrid search with optional cross-encoder reranking, metadata filtering
3. **Prompt Builder** (`prompt_builder.py`) - Constructs academic-style prompts with citation enforcement
4. **API Endpoint** (`api.py`) - FastAPI server with `/ask` endpoint and Web UI

## Installation

### Option 1: Docker Setup (Recommended)

```bash
cp env.example .env
# Edit .env and set OPENAI_API_KEY
bash start.bash
```

This will start Docker (if needed), build images, and start all services (Qdrant, API, vLLM if configured).

**See [DOCKER.md](DOCKER.md) for complete Docker documentation.**

### Option 2: Local Python Setup

```bash
pip install -r requirements.txt
cp env.example .env
# Edit .env and set OPENAI_API_KEY

# Set up Qdrant (Docker recommended)
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### Paper Format

Papers should be in JSON format:

```json
{
  "paper_id": "narayanan2020",
  "title": "Proxemic Fusion: Emotion Recognition from Gait",
  "authors": ["Narayanan, A.", "Smith, B."],
  "year": 2020,
  "citation_pointer": "Narayanan et al., 2020",
  "modality_tags": ["gait", "emotion_recognition"],
  "sections": [
    {
      "section_name": "I. Introduction",
      "text": "Emotion recognition from gait patterns...",
      "page_start": 1,
      "page_end": 2
    }
  ]
}
```

### Query the System

**Web UI:** http://localhost:8000/

**API:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does emotion affect gait patterns?",
    "top_k": 8,
    "year_min": 2018
  }'
```

**API with Quality Evaluation:**
```bash
curl -X POST "http://localhost:8000/ask-with-evaluation" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does emotion affect gait patterns?",
    "top_k": 8
  }' | jq .
```

This endpoint returns the same answer and sources as `/ask`, plus comprehensive quality metrics:
- **Faithfulness (0-1)**: Are all claims supported by retrieved evidence?
- **Coverage (0-2)**: Did it cover key papers? (only when ground truth provided)
- **Utility (0-2)**: Is it publication-ready?
- **Citation diversity ratio**: Ratio of cited papers to available papers (informational)
- **Validation results**: Citation verification, hallucination detection, confidence scores

**Python:**
```python
import requests
response = requests.post("http://localhost:8000/ask", json={
    "question": "Explain proxemic fusion",
    "top_k": 10,
    "modality_tags": ["gait"]
})
print(response.json()["answer"])
```

## API Endpoints

- `POST /ask` - Ask questions about the literature
- `POST /ask-with-evaluation` - Ask questions with quality metrics and validation
- `POST /retrieve` - Retrieve chunks without generating answer
- `GET /health` - Health check
- `GET /stats` - Collection statistics
- `GET /` - Web UI
- `GET /api` - API documentation

**Request body for `/ask` and `/ask-with-evaluation`:**
```json
{
  "question": "string (required)",
  "top_k": 8,
  "year_min": 2018,
  "year_max": 2024,
  "modality_tags": ["gait"],
  "use_reranking": true,
  "rerank_initial_k": 20
}
```

**Response from `/ask-with-evaluation` includes:**
```json
{
  "answer": "...",
  "sources": [...],
  "question": "...",
  "quality_metrics": {
    "faithfulness": 0.85,
    "coverage": null,
    "utility": 1.8,
    "citation_diversity_ratio": 0.5,
    "papers_cited_count": 4,
    "papers_available_count": 8,
    "hallucination_count": 0,
    "unsupported_claims_count": 1
  },
  "validation": {
    "is_valid": true,
    "confidence": 0.92,
    "citations_total": 5,
    "citations_verified": 5
  }
}
```

## Configuration

All settings in `config.py` can be overridden via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/vllm) | `vllm` |
| `LLM_MODEL` | Model name | `mistral-7b-fp8` |
| `EMBEDDING_MODEL` | Sentence transformer model | `BAAI/bge-base-en-v1.5` |
| `USE_HYBRID_SEARCH` | Enable dense + sparse search | `True` |
| `USE_RERANKING` | Enable cross-encoder reranking | `True` |
| `RERANKER_MODEL` | Cross-encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CHUNK_SIZE` | Target chunk size (tokens) | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks (tokens) | `200` |
| `DEFAULT_TOP_K` | Default retrieval count | `8` |

## Project Structure

```
academic-rag/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration management
├── ingest.py                 # Ingestion script
├── retriever.py              # Retrieval logic
├── prompt_builder.py         # Prompt construction
├── llm_client.py             # LLM client abstraction
├── api.py                    # FastAPI server
├── docker-compose.yml        # Docker orchestration
├── static/                   # Web UI files
└── extracted_data/           # Your paper JSON files
```

## Troubleshooting

- **"Collection not found"** - Run the ingest script first
- **"No relevant literature found"** - Check papers are ingested, try broader question
- **"Error calling LLM"** - Verify API key (OpenAI) or vLLM server is running
- **Embeddings are slow** - Use smaller model or enable GPU acceleration
- **Citations not showing** - Check `citation_pointer` in paper JSON, lower LLM temperature

## License

MIT License - Use freely for academic research.

---

**Built for researchers who need control, not convenience.**
