# vLLM Model Compression Guide

This guide explains how to use the `vllm_compress.py` script to compress LLMs for optimized deployment with vLLM.

## Installation

First, install the required dependencies:

```bash
pip install llmcompressor vllm transformers torch
```

## Quick Start

### Basic Usage

Compress a model with default settings (INT8 weights and activations):

```bash
python vllm_compress.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir ./compressed_models/tinyllama-int8
```

### Available Quantization Schemes

1. **w8a8_int8** (default): INT8 weights and activations using GPTQ + SmoothQuant
   - Best balance of speed and accuracy
   - ~2x memory reduction
   
2. **w8a8_fp8**: FP8 weights and activations
   - Better accuracy than INT8
   - Requires modern GPU support (H100, etc.)
   
3. **w4a16_gptq**: INT4 weights with FP16 activations using GPTQ
   - ~4x memory reduction
   - Good for memory-constrained scenarios
   
4. **w4a16_awq**: INT4 weights with FP16 activations using AWQ
   - Alternative to GPTQ
   - Often better accuracy retention
   
5. **w8a16**: INT8 weights with FP16 activations
   - ~2x memory reduction
   - Simpler than W8A8
   
6. **w4a4_fp4**: FP4 weights and activations (NVFP4)
   - ~4x memory and speed improvement
   - Requires latest GPU support

## Examples

### Example 1: Compress Llama-2 with FP8

```bash
python vllm_compress.py \
  --model meta-llama/Llama-2-7b-hf \
  --output-dir ./compressed_models/llama2-7b-fp8 \
  --scheme w8a8_fp8 \
  --num-calibration-samples 256
```

### Example 2: Compress Mistral with 4-bit GPTQ

```bash
python vllm_compress.py \
  --model mistralai/Mistral-7B-v0.1 \
  --output-dir ./compressed_models/mistral-7b-w4a16 \
  --scheme w4a16_gptq \
  --max-seq-length 4096
```

### Example 3: Compress with Custom Dataset

```bash
python vllm_compress.py \
  --model facebook/opt-1.3b \
  --output-dir ./compressed_models/opt-1.3b-int8 \
  --scheme w8a8_int8 \
  --dataset wikitext \
  --num-calibration-samples 128
```

### Example 4: Use Trust Remote Code (for some models)

```bash
python vllm_compress.py \
  --model Qwen/Qwen2-7B \
  --output-dir ./compressed_models/qwen2-7b-int8 \
  --scheme w8a8_int8 \
  --trust-remote-code
```

### Example 5: Compress and Push to HuggingFace Hub

```bash
python vllm_compress.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir ./compressed_models/tinyllama-int8 \
  --scheme w8a8_int8 \
  --push-to-hub \
  --hub-repo-name tinyllama-1.1b-int8
```

### Example 6: Push to Organization on HuggingFace Hub

```bash
python vllm_compress.py \
  --model meta-llama/Llama-2-7b-hf \
  --output-dir ./compressed_models/llama2-7b-fp8 \
  --scheme w8a8_fp8 \
  --push-to-hub \
  --hub-repo-name llama2-7b-fp8 \
  --hub-organization my-org \
  --hub-token hf_xxxxxxxxxxxxx
```

## Using the Compressed Model in vLLM

After compression, the model is automatically saved to the specified output directory by the `oneshot()` function. You can use it with vLLM:

### Command Line

```bash
vllm serve ./compressed_models/tinyllama-int8
```

Or with Docker:

```bash
docker run --gpus all -v $(pwd)/compressed_models/tinyllama-int8:/models/tinyllama-int8:ro \
  -p 8000:8000 vllm/vllm-openai:latest \
  --model /models/tinyllama-int8
```

### Python API

```python
from vllm import LLM

# Load the compressed model
model = LLM("./compressed_models/tinyllama-int8")

# Generate text
output = model.generate("Once upon a time")
print(output)
```

## Parameters

- `--model`: HuggingFace model name or local path (required)
- `--output-dir`: Directory to save compressed model (required)
- `--scheme`: Quantization scheme (default: w8a8_int8)
  - Available: `w8a8_int8`, `w8a8_fp8`, `w4a16_gptq`, `w4a16_awq`, `w8a16`, `w4a4_fp4`
- `--dataset`: Calibration dataset name or path (default: open_platypus)
- `--max-seq-length`: Max sequence length for calibration (default: 2048)
- `--num-calibration-samples`: Number of calibration samples (default: 512)
- `--trust-remote-code`: Trust remote code in model (flag)
- `--push-to-hub`: Push compressed model to HuggingFace Hub after compression (flag)
- `--hub-repo-name`: Repository name on HuggingFace Hub (required if --push-to-hub is set)
- `--hub-organization`: Organization name on HuggingFace Hub (optional)
- `--hub-token`: HuggingFace authentication token (optional, uses CLI login if not provided)

## Advanced Usage

### Using as a Python Module

The script can be imported and used programmatically:

```python
import sys
sys.path.append('vllm_compression')  # Add path if needed

from vllm_compress import VLLMCompressor

# Create compressor
compressor = VLLMCompressor(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    output_dir="./compressed_models/custom",
    quantization_scheme="w8a8_int8",
    num_calibration_samples=256
)

# Run compression
compressor.compress()
```

### Push to HuggingFace Hub Programmatically

```python
from vllm_compress import VLLMCompressor

compressor = VLLMCompressor(
    model_name="meta-llama/Llama-2-7b-hf",
    output_dir="./compressed_models/llama2-7b-fp8",
    quantization_scheme="w8a8_fp8",
    push_to_hub=True,
    hub_repo_name="llama2-7b-fp8",
    hub_organization="my-org",  # Optional
    hub_token="hf_xxxxxxxxxxxxx"  # Optional, uses CLI login if not provided
)

compressor.compress()
```

### Custom Recipe

You can define a custom compression recipe instead of using a predefined scheme:

```python
import sys
sys.path.append('vllm_compression')  # Add path if needed

from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from vllm_compress import VLLMCompressor

# Define custom recipe
custom_recipe = [
    SmoothQuantModifier(smoothing_strength=0.5),
    GPTQModifier(
        scheme="W8A8",
        targets="Linear",
        ignore=["lm_head", "embed_tokens"]
    ),
]

# Use custom recipe (custom_recipe overrides quantization_scheme)
compressor = VLLMCompressor(
    model_name="meta-llama/Llama-2-7b-hf",
    output_dir="./compressed_models/custom",
    custom_recipe=custom_recipe
)
compressor.compress()
```

## How It Works

The script uses the `llm-compressor` library's `oneshot()` function, which:

1. Loads the model and tokenizer from HuggingFace or local path
2. Applies the specified compression recipe (quantization, etc.)
3. **Automatically saves the compressed model** to the output directory
4. Enhances `model.save_pretrained()` with compression support for future saves

The compressed model is ready to use with vLLM immediately after compression completes.

## Tips and Best Practices

1. **Start with fewer calibration samples** for testing, then increase for production
2. **Use FP8** if you have H100 or newer GPUs for best speed/accuracy tradeoff
3. **Use W4A16** for maximum memory savings when GPU memory is limited
4. **Adjust max-seq-length** based on your use case (longer for long-context models)
5. **Test accuracy** on your specific tasks after compression
6. **Monitor GPU memory** during compression (large models may need significant RAM)
7. **Use local paths** for models you've already downloaded to avoid re-downloading
8. **Push to HuggingFace Hub** to share compressed models or use them across different machines

## Troubleshooting

### Out of Memory
- Reduce `--num-calibration-samples`
- Reduce `--max-seq-length`
- Use a machine with more RAM/VRAM

### Model Not Loading
- Add `--trust-remote-code` flag
- Check model compatibility with transformers version
- Verify model name/path is correct

### Poor Accuracy After Compression
- Increase `--num-calibration-samples`
- Try different quantization scheme (e.g., w8a8_fp8 instead of w8a8_int8)
- Use a more representative calibration dataset

### HuggingFace Hub Push Fails
- Ensure you're logged in: `huggingface-cli login` or provide `--hub-token`
- Verify repository name is available and you have write access
- Check that `--hub-repo-name` is provided when using `--push-to-hub`
- For organization repos, ensure you have permission to create repos in that org

## Resources

- [llm-compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [llm-compressor Documentation](https://docs.vllm.ai/projects/llm-compressor)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Compression Schemes Guide](https://github.com/vllm-project/llm-compressor/blob/main/docs/compression_schemes.md)

