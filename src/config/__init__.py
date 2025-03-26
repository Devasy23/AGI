from .llm_config import LLMConfig
from .memory_config import MemoryConfig

class Config:
    @staticmethod
    def get_all():
        """Get all configuration settings"""
        return {
            "llm": LLMConfig.get_config(),
            "memory": MemoryConfig.get_config()
        }
    
    @staticmethod
    def validate_all():
        """Validate all configurations"""
        # Will raise ValueError if validation fails
        MemoryConfig.validate_config()

__all__ = ['LLMConfig', 'MemoryConfig', 'Config']