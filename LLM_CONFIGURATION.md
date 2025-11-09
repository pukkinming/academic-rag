# LLM Configuration Guide

This guide explains how to configure and switch between different LLM providers (OpenAI and vLLM) in the Emotion Recognition RAG system.

## Overview

The system now supports two LLM providers:

1. **OpenAI API** - Cloud-based models (GPT-4, GPT-3.5-turbo, etc.)
2. **vLLM** - Self-hosted models with OpenAI-compatible API

You can easily switch between them by changing configuration in your `.env` file.

## Quick Start

### Option 1: Using OpenAI API

1. **Create/edit your `.env` file:**
```bash
cp env.example .env
```

2. **Configure for OpenAI:**
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
OPENAI_API_KEY=your-openai-api-key-here
```

3. **Start the server:**
```bash
python api.py
```

### Option 2: Using vLLM (Self-Hosted)

1. **Start your vLLM server** (in a separate terminal):
```bash
# Example: Running Mistral-7B with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --host 0.0.0.0 \
    --port 8000
```

Or using Docker:
```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.2
```

2. **Configure for vLLM** (edit `.env`):
```bash
LLM_PROVIDER=vllm
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
```

3. **Start the RAG API server** (use a different port):
```bash
# Edit .env to use a different port for the RAG API
API_PORT=8001

# Then start
python api.py
```

Now your RAG API will be on port 8001, and it will call vLLM on port 8000.

## Detailed Configuration

### Environment Variables

#### Core LLM Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LLM_PROVIDER` | Provider to use (`openai` or `vllm`) | `openai` | Yes |
| `LLM_MODEL` | Model name/identifier | `gpt-4-turbo-preview` | Yes |
| `LLM_TEMPERATURE` | Sampling temperature (0.0-2.0) | `0.1` | No |
| `LLM_MAX_TOKENS` | Maximum tokens to generate | `2000` | No |

#### OpenAI-Specific Settings

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes (for OpenAI) |
| `OPENAI_BASE_URL` | Custom endpoint (for Azure, etc.) | No |

#### vLLM-Specific Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VLLM_BASE_URL` | vLLM server endpoint | `http://localhost:8000/v1` | Yes (for vLLM) |
| `VLLM_API_KEY` | API key (if required) | `EMPTY` | No |

### Example Configurations

#### Example 1: OpenAI GPT-4
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
```

#### Example 2: OpenAI GPT-3.5 (cheaper, faster)
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
```

#### Example 3: Azure OpenAI
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your-azure-api-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
```

#### Example 4: vLLM with Mistral-7B
```bash
LLM_PROVIDER=vllm
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
```

#### Example 5: vLLM with Llama-2-70B
```bash
LLM_PROVIDER=vllm
LLM_MODEL=meta-llama/Llama-2-70b-chat-hf
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
```

#### Example 6: Remote vLLM Server
```bash
LLM_PROVIDER=vllm
LLM_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
VLLM_BASE_URL=http://gpu-server.example.com:8000/v1
VLLM_API_KEY=your-secret-key
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000
```

## Setting Up vLLM

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+
- 16GB+ GPU memory (depends on model size)

### Installation

```bash
# Install vLLM
pip install vllm

# Or with GPU support
pip install vllm[cuda]
```

### Running vLLM Server

#### Basic Usage
```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --host 0.0.0.0 \
    --port 8000
```

#### With Advanced Options
```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

#### Using Docker
```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=your-token" \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization 0.9
```

### Testing vLLM Server

```bash
# Check if vLLM is running
curl http://localhost:8000/v1/models

# Test generation
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }'
```

## Testing Your Configuration

### 1. Test LLM Client Directly
```bash
python llm_client.py
```

### 2. Check API Health
```bash
curl http://localhost:8000/health
```

### 3. Get LLM Info
```bash
curl http://localhost:8000/llm-info
```

### 4. Test Question Answering
```bash
curl -X POST "http://localhost:8000/ask" \
    -H "Content-Type: application/json" \
    -d '{
        "question": "How does emotion affect gait patterns?",
        "top_k": 5
    }'
```

## Switching Between Providers

To switch providers, simply update your `.env` file and restart the server:

### From OpenAI to vLLM:
1. Start vLLM server
2. Edit `.env`:
   ```bash
   LLM_PROVIDER=vllm
   LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
   VLLM_BASE_URL=http://localhost:8000/v1
   ```
3. Restart API server: `python api.py`

### From vLLM to OpenAI:
1. Edit `.env`:
   ```bash
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4-turbo-preview
   OPENAI_API_KEY=your-key
   ```
2. Restart API server: `python api.py`

## Recommended Models

### For OpenAI:
- **GPT-4 Turbo** (`gpt-4-turbo-preview`) - Best quality, most expensive
- **GPT-4** (`gpt-4`) - Excellent quality, expensive
- **GPT-3.5 Turbo** (`gpt-3.5-turbo`) - Good quality, very cheap

### For vLLM (Self-Hosted):
- **Mistral-7B** (`mistralai/Mistral-7B-Instruct-v0.2`) - Great quality/speed ratio, ~14GB VRAM
- **Llama-2-13B** (`meta-llama/Llama-2-13b-chat-hf`) - Solid quality, ~26GB VRAM
- **Mixtral-8x7B** (`mistralai/Mixtral-8x7B-Instruct-v0.1`) - Excellent quality, ~90GB VRAM
- **Llama-2-70B** (`meta-llama/Llama-2-70b-chat-hf`) - Best open-source, ~140GB VRAM

## Performance Comparison

| Provider | Model | Typical Latency | Cost per 1K tokens | Quality |
|----------|-------|----------------|-------------------|---------|
| OpenAI | GPT-4 Turbo | 2-5s | $0.01-0.03 | ★★★★★ |
| OpenAI | GPT-3.5 Turbo | 1-2s | $0.001-0.002 | ★★★★ |
| vLLM | Mistral-7B | 1-3s | Free* | ★★★★ |
| vLLM | Mixtral-8x7B | 2-5s | Free* | ★★★★★ |
| vLLM | Llama-2-70B | 3-8s | Free* | ★★★★★ |

*Free after GPU infrastructure costs

## Troubleshooting

### OpenAI Issues

**"Authentication failed"**
- Check that `OPENAI_API_KEY` is correct
- Verify you have credits in your OpenAI account
- Check the key hasn't expired

**"Rate limit exceeded"**
- You're hitting OpenAI's rate limits
- Wait a few minutes or upgrade your plan
- Consider using vLLM for higher throughput

### vLLM Issues

**"Connection refused"**
- Make sure vLLM server is running: `curl http://localhost:8000/v1/models`
- Check the port in `VLLM_BASE_URL` matches your vLLM server
- Verify firewall settings

**"CUDA out of memory"**
- Your model is too large for your GPU
- Try a smaller model (e.g., Mistral-7B instead of Llama-2-70B)
- Reduce `--gpu-memory-utilization` (e.g., 0.8)
- Use `--tensor-parallel-size` for multi-GPU

**"Model not found"**
- Check the model name exactly matches (case-sensitive)
- Verify you have access to the model on HuggingFace
- For gated models, set `HUGGING_FACE_HUB_TOKEN`

## Best Practices

1. **Development**: Use GPT-3.5-turbo (cheap, fast) or Mistral-7B (free, fast)
2. **Production**: Use GPT-4-turbo (quality) or Mixtral-8x7B (cost-effective)
3. **Low Latency**: Use vLLM with smaller models (Mistral-7B)
4. **High Throughput**: Use vLLM with batching enabled
5. **Citations**: Lower temperature (0.1-0.3) for more consistent citations
6. **Creative Tasks**: Higher temperature (0.7-1.0) for more diversity

## Cost Optimization

### OpenAI Cost Tips:
- Use `gpt-3.5-turbo` for most queries (~100x cheaper than GPT-4)
- Reserve GPT-4 for complex questions
- Set reasonable `max_tokens` limits
- Cache common queries

### vLLM Cost Tips:
- Share one vLLM server across multiple RAG instances
- Use tensor parallelism to serve bigger models efficiently
- Batch multiple queries together
- Use continuous batching (`--enable-continuous-batching`)

## Advanced: Multiple LLM Backends

You can run multiple configurations simultaneously by using different environment files:

```bash
# Start OpenAI version on port 8001
LLM_PROVIDER=openai python api.py --port 8001

# Start vLLM version on port 8002
LLM_PROVIDER=vllm python api.py --port 8002
```

Or use Docker Compose with different service definitions.

## Need Help?

- Check `/health` endpoint for system status
- Check `/llm-info` endpoint for current LLM configuration
- Review logs for detailed error messages
- Test LLM directly with `python llm_client.py`

---

**Built for flexibility: Choose the right LLM for your needs.**

