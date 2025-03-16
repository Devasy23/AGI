from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMInterface(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], format: str = "json") -> Dict[str, Any]:
        """Execute a chat completion with the LLM"""
        pass
    
    @abstractmethod
    def prepare_prompt(self, system_prompt: str, user_query: str, context: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """Prepare the messages for a chat completion"""
        pass