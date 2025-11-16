#!/usr/bin/env python3
"""
vLLM Model Compression Script using llm-compressor

This script applies various compression algorithms to LLMs for optimized deployment with vLLM.
Supports multiple quantization methods including:
- W8A8 (int8 and fp8)
- W4A16, W8A16
- NVFP4
- GPTQ, AWQ, SmoothQuant algorithms

Reference: https://github.com/vllm-project/llm-compressor
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List

from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VLLMCompressor:
    """Handler for compressing LLMs using llm-compressor."""
    
    QUANTIZATION_SCHEMES = {
        'w8a8_int8': {
            'description': 'INT8 weights and activations (GPTQ + SmoothQuant)',
            'recipe': lambda: [
                SmoothQuantModifier(smoothing_strength=0.8),
                GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
            ]
        },
        'w8a8_fp8': {
            'description': 'FP8 weights and activations',
            'recipe': lambda: [
                QuantizationModifier(
                    targets="Linear",
                    scheme="FP8",
                    ignore=["lm_head"]
                )
            ]
        },
        'w4a16_gptq': {
            'description': 'INT4 weights with FP16 activations using GPTQ',
            'recipe': lambda: [
                GPTQModifier(
                    targets="Linear",
                    scheme="W4A16",
                    ignore=["lm_head"]
                )
            ]
        },
        'w4a16_awq': {
            'description': 'INT4 weights with FP16 activations using AWQ',
            'recipe': lambda: [
                QuantizationModifier(
                    targets="Linear",
                    scheme="W4A16",
                    ignore=["lm_head"],
                    config_groups={
                        "group_0": {
                            "targets": ["Linear"],
                            "input_activations": None,
                            "weights": {
                                "num_bits": 4,
                                "type": "int",
                                "symmetric": True,
                                "strategy": "channel",
                            }
                        }
                    }
                )
            ]
        },
        'w8a16': {
            'description': 'INT8 weights with FP16 activations',
            'recipe': lambda: [
                GPTQModifier(
                    targets="Linear",
                    scheme="W8A16",
                    ignore=["lm_head"]
                )
            ]
        },
        'w4a4_fp4': {
            'description': 'FP4 weights and activations (NVFP4)',
            'recipe': lambda: [
                QuantizationModifier(
                    targets="Linear",
                    scheme="NVFP4",
                    ignore=["lm_head"]
                )
            ]
        }
    }
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        quantization_scheme: str = 'w8a8_int8',
        dataset: str = 'open_platypus',
        max_seq_length: int = 2048,
        num_calibration_samples: int = 512,
        trust_remote_code: bool = False,
        custom_recipe: Optional[List] = None,
        push_to_hub: bool = False,
        hub_repo_name: Optional[str] = None,
        hub_organization: Optional[str] = None,
        hub_token: Optional[str] = None
    ):
        """
        Initialize the VLLMCompressor.
        
        Args:
            model_name: HuggingFace model name or local path
            output_dir: Directory to save compressed model
            quantization_scheme: Quantization scheme to use
            dataset: Calibration dataset name or path
            max_seq_length: Maximum sequence length for calibration
            num_calibration_samples: Number of calibration samples
            trust_remote_code: Whether to trust remote code in model
            custom_recipe: Optional custom recipe (overrides quantization_scheme)
            push_to_hub: Whether to push compressed model to HuggingFace Hub
            hub_repo_name: Repository name on HuggingFace Hub
            hub_organization: Optional organization name for HuggingFace Hub
            hub_token: Optional HuggingFace token (uses CLI login if not provided)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.quantization_scheme = quantization_scheme
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.num_calibration_samples = num_calibration_samples
        self.trust_remote_code = trust_remote_code
        self.custom_recipe = custom_recipe
        self.push_to_hub = push_to_hub
        self.hub_repo_name = hub_repo_name
        self.hub_organization = hub_organization
        self.hub_token = hub_token
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_recipe(self) -> List:
        """Get the quantization recipe."""
        if self.custom_recipe:
            logger.info("Using custom recipe")
            return self.custom_recipe
            
        if self.quantization_scheme not in self.QUANTIZATION_SCHEMES:
            available = ', '.join(self.QUANTIZATION_SCHEMES.keys())
            raise ValueError(
                f"Unknown quantization scheme: {self.quantization_scheme}. "
                f"Available schemes: {available}"
            )
        
        scheme_info = self.QUANTIZATION_SCHEMES[self.quantization_scheme]
        logger.info(f"Using scheme: {self.quantization_scheme} - {scheme_info['description']}")
        return scheme_info['recipe']()
    
    def compress(self):
        """Apply compression to the model."""
        logger.info(f"Starting compression of model: {self.model_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Calibration dataset: {self.dataset}")
        logger.info(f"Max sequence length: {self.max_seq_length}")
        logger.info(f"Number of calibration samples: {self.num_calibration_samples}")
        
        recipe = self.get_recipe()
        
        try:
            # Load model and tokenizer
            logger.info("Loading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=self.trust_remote_code
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            # Apply compression using oneshot
            # This automatically enhances model.save_pretrained with compression capabilities
            logger.info("Applying compression...")
            oneshot(
                model=model,
                dataset=self.dataset,
                recipe=recipe,
                max_seq_length=self.max_seq_length,
                num_calibration_samples=self.num_calibration_samples,
                output_dir=self.output_dir,
                # skip_sparsity_compression_stats=True,
            )
            
            logger.info(f"✓ Compression completed successfully!")
            
            # # Save compressed model locally using enhanced save_pretrained
            # logger.info(f"Saving compressed model to: {self.output_dir}")
            # model.save_pretrained(
            #     str(self.output_dir),
            #     save_compressed=True
            # )
            # tokenizer.save_pretrained(str(self.output_dir))
            
            logger.info(f"✓ Compressed model saved to: {self.output_dir}")
            logger.info(f"✓ You can now load this model in vLLM")
            
            # Push to HuggingFace Hub if requested
            if self.push_to_hub:
                self._push_to_hub(model, tokenizer)
            
        except Exception as e:
            logger.error(f"✗ Compression failed: {e}")
            raise
    
    def _push_to_hub(self, model, tokenizer):
        """Push the compressed model to HuggingFace Hub using the enhanced save_pretrained.
        
        Args:
            model: The compressed model with enhanced save_pretrained
            tokenizer: The model's tokenizer
        """
        if not self.hub_repo_name:
            logger.error("✗ Hub repository name is required when push_to_hub is enabled")
            raise ValueError("hub_repo_name must be specified when push_to_hub=True")
        
        logger.info(f"Pushing compressed model to HuggingFace Hub...")
        
        try:
            # Prepare repo name with organization if specified
            repo_id = self.hub_repo_name
            if self.hub_organization:
                repo_id = f"{self.hub_organization}/{self.hub_repo_name}"
            
            logger.info(f"Repository: {repo_id}")
            
            # Push model to hub using enhanced save_pretrained
            # After oneshot(), save_pretrained is enhanced with compression support
            logger.info(f"Uploading compressed model to {repo_id}...")
            model.save_pretrained(
                repo_id,
                save_compressed=True,
                push_to_hub=True,
                token=self.hub_token
            )
            
            # Push tokenizer to hub
            logger.info(f"Uploading tokenizer to {repo_id}...")
            tokenizer.save_pretrained(
                repo_id,
                push_to_hub=True,
                token=self.hub_token
            )
            
            logger.info(f"✓ Successfully pushed compressed model to HuggingFace Hub: {repo_id}")
            logger.info(f"✓ Model available at: https://huggingface.co/{repo_id}")
            
        except Exception as e:
            logger.error(f"✗ Failed to push to HuggingFace Hub: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Compress LLMs for vLLM deployment using llm-compressor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress TinyLlama with INT8 weights and activations
  python vllm_compress.py \\
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --output-dir ./compressed_models/tinyllama-int8 \\
    --scheme w8a8_int8

  # Compress with FP8 quantization
  python vllm_compress.py \\
    --model meta-llama/Llama-2-7b-hf \\
    --output-dir ./compressed_models/llama2-7b-fp8 \\
    --scheme w8a8_fp8 \\
    --num-calibration-samples 256

  # Compress with 4-bit weights using GPTQ
  python vllm_compress.py \\
    --model mistralai/Mistral-7B-v0.1 \\
    --output-dir ./compressed_models/mistral-7b-w4a16 \\
    --scheme w4a16_gptq

  # Compress and push to HuggingFace Hub
  python vllm_compress.py \\
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --output-dir ./compressed_models/tinyllama-int8 \\
    --scheme w8a8_int8 \\
    --push-to-hub \\
    --hub-repo-name tinyllama-1.1b-int8

  # Compress and push to organization on HuggingFace Hub
  python vllm_compress.py \\
    --model meta-llama/Llama-2-7b-hf \\
    --output-dir ./compressed_models/llama2-7b-fp8 \\
    --scheme w8a8_fp8 \\
    --push-to-hub \\
    --hub-repo-name llama2-7b-fp8 \\
    --hub-organization my-org

Available quantization schemes:
  - w8a8_int8: INT8 weights and activations (GPTQ + SmoothQuant)
  - w8a8_fp8: FP8 weights and activations
  - w4a16_gptq: INT4 weights with FP16 activations using GPTQ
  - w4a16_awq: INT4 weights with FP16 activations using AWQ
  - w8a16: INT8 weights with FP16 activations
  - w4a4_fp4: FP4 weights and activations (NVFP4)
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='HuggingFace model name or local path to model'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save the compressed model'
    )
    
    parser.add_argument(
        '--scheme',
        type=str,
        default='w8a8_int8',
        choices=list(VLLMCompressor.QUANTIZATION_SCHEMES.keys()),
        help='Quantization scheme to use (default: w8a8_int8)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='open_platypus',
        help='Calibration dataset name or path (default: open_platypus)'
    )
    
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=2048,
        help='Maximum sequence length for calibration (default: 2048)'
    )
    
    parser.add_argument(
        '--num-calibration-samples',
        type=int,
        default=512,
        help='Number of calibration samples (default: 512)'
    )
    
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Trust remote code in model (required for some models)'
    )
    
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push compressed model to HuggingFace Hub after compression'
    )
    
    parser.add_argument(
        '--hub-repo-name',
        type=str,
        help='Repository name on HuggingFace Hub (required if --push-to-hub is set)'
    )
    
    parser.add_argument(
        '--hub-organization',
        type=str,
        help='Organization name on HuggingFace Hub (optional)'
    )
    
    parser.add_argument(
        '--hub-token',
        type=str,
        help='HuggingFace authentication token (optional, uses CLI login if not provided)'
    )
    
    args = parser.parse_args()
    
    # Create compressor and run
    compressor = VLLMCompressor(
        model_name=args.model,
        output_dir=args.output_dir,
        quantization_scheme=args.scheme,
        dataset=args.dataset,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
        trust_remote_code=args.trust_remote_code,
        push_to_hub=args.push_to_hub,
        hub_repo_name=args.hub_repo_name,
        hub_organization=args.hub_organization,
        hub_token=args.hub_token
    )
    
    compressor.compress()


if __name__ == '__main__':
    main()

