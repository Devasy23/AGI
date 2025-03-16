from typing import Dict, Type
from .llm_interface import LLMInterface
from .ollama_llm import OllamaLLM
from .groq_llm import GroqLLM
from .gemini_llm import GeminiLLM
from ..config.llm_config import LLMConfig

class LLMFactory:
    _providers: Dict[str, Type[LLMInterface]] = {
        "ollama": OllamaLLM,
        "groq": GroqLLM,
        "gemini": GeminiLLM
    }
    
    @classmethod
    def create_llm(cls) -> LLMInterface:
        """Create an LLM instance based on the configuration"""
        config = LLMConfig.get_config()
        provider = config["provider"]
        
        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return cls._providers[provider]()