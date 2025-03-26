from crewai import LLM
from src.config.llm_config import LLMConfig

def create_llm():
    """Create a CrewAI LLM instance based on configuration"""
    config = LLMConfig.get_config()
    provider = config["provider"]
    model_name = config["model_name"]
    
    if provider == "ollama":
        return LLM(
            model=f"ollama/{model_name}",
            base_url="http://localhost:11434"
        )
    elif provider == "groq":
        return LLM(
            model=f"groq/{model_name}",
            api_key=config.get("groq_api_key")
        )
    elif provider == "gemini":
        vertex_credentials = config.get("vertex_credentials")  # Should be configured in llm_config.py
        return LLM(
            model=f"gemini/{model_name}",
            temperature=0.7,
            vertex_credentials=vertex_credentials
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# Export the create_llm function as the main interface
__all__ = ['create_llm']