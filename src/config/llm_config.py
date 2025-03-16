from typing import Literal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMConfig:
    provider: Literal["ollama", "groq", "gemini"] = os.getenv("LLM_PROVIDER", "ollama")
    model_name: str = os.getenv("LLM_MODEL", "gemma3:4b")
    
    # API Keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    @classmethod
    def get_config(cls):
        return {
            "provider": cls.provider,
            "model_name": cls.model_name,
            "groq_api_key": cls.groq_api_key,
            "gemini_api_key": cls.gemini_api_key
        }