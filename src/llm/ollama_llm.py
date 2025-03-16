import ollama
from typing import List, Dict, Any
from .llm_interface import LLMInterface
from ..config.llm_config import LLMConfig

class OllamaLLM(LLMInterface):
    def __init__(self):
        self.config = LLMConfig.get_config()
        self.model = self.config["model_name"]
    
    def chat(self, messages: List[Dict[str, str]], format: str = "json") -> Dict[str, Any]:
        return ollama.chat(model=self.model, messages=messages, format=format)
    
    def prepare_prompt(self, system_prompt: str, user_query: str, context: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_query})
        return messages