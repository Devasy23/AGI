from .llm_config import LLMConfig
from .memory_config import MemoryConfig
from .mcp_config import MCPConfig

class Config:
    @staticmethod
    def get_all():
        """Get all configuration settings"""
        return {
            "llm": LLMConfig.get_config(),
            "memory": MemoryConfig.get_config(),
            "mcp": MCPConfig.get_config()
        }
    
    @staticmethod
    def validate_all():
        """Validate all configurations"""
        # Will raise ValueError if validation fails
        MemoryConfig.validate_config()
        MCPConfig.validate_config()

__all__ = ['LLMConfig', 'MemoryConfig', 'MCPConfig', 'Config']