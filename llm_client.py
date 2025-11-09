"""LLM client abstraction supporting OpenAI and vLLM."""

from typing import List, Dict, Any
from openai import OpenAI
from config import settings


class LLMClient:
    """
    Unified LLM client that supports multiple providers.
    
    Supports:
    - OpenAI API (gpt-4, gpt-3.5-turbo, etc.)
    - vLLM (OpenAI-compatible API for local models)
    """
    
    def __init__(self):
        """Initialize the LLM client based on configured provider."""
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "vllm":
            self._init_vllm()
        else:
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}. "
                f"Supported providers: 'openai', 'vllm'"
            )
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
        
        # Support custom base URLs (e.g., Azure OpenAI)
        client_kwargs = {"api_key": settings.openai_api_key}
        # Only set base_url if it's not None (config converts empty strings to None)
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        
        self.client = OpenAI(**client_kwargs)
        print(f"✓ Initialized OpenAI client with model: {self.model}")
    
    def _init_vllm(self):
        """Initialize vLLM client (OpenAI-compatible API)."""
        if not settings.vllm_base_url:
            raise ValueError(
                "vLLM base URL not found. Please set VLLM_BASE_URL environment variable."
            )
        
        # vLLM uses OpenAI-compatible API
        self.client = OpenAI(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key or "EMPTY"  # vLLM typically doesn't need a key
        )
        print(f"✓ Initialized vLLM client at {settings.vllm_base_url} with model: {self.model}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens to generate (overrides default)
        
        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            error_type = type(e).__name__
            error_details = str(e)
            # For connection errors, try to get more details
            if hasattr(e, '__cause__') and e.__cause__:
                error_details += f" | Cause: {str(e.__cause__)}"
            error_msg = f"Error calling {self.provider.upper()} API: {error_type}: {error_details}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def generate_streaming(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (overrides default)
            max_tokens: Max tokens to generate (overrides default)
        
        Yields:
            Text chunks as they are generated
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            error_msg = f"Error streaming from {self.provider.upper()} API: {str(e)}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM configuration.
        
        Returns:
            Dictionary with provider, model, and settings
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": (
                settings.openai_base_url if self.provider == "openai" 
                else settings.vllm_base_url
            )
        }


# Global client instance (lazy initialization)
_llm_client = None


def get_llm_client() -> LLMClient:
    """
    Get or create the global LLM client instance.
    
    Returns:
        LLMClient instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


if __name__ == "__main__":
    # Test the LLM client
    print("Testing LLM Client...")
    print("=" * 80)
    
    client = get_llm_client()
    print(f"\nClient info: {client.get_info()}")
    
    # Test simple generation
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."}
    ]
    
    print("\nTesting generation...")
    try:
        response = client.generate(test_messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print("Test complete!")

