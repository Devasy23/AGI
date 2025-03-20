import groq
import json
from typing import List, Dict, Any
from .llm_interface import LLMInterface
from src.config.llm_config import LLMConfig

class GroqLLM(LLMInterface):
    def __init__(self):
        self.config = LLMConfig.get_config()
        self.client = groq.Groq(api_key=self.config["groq_api_key"])
    
    def chat(self, messages: List[Dict[str, str]], format: str = "json") -> Dict[str, Any]:
        completion = self.client.chat.completions.create(
            model=self.config["model_name"],  # Using Mixtral model
            messages=messages,
            response_format={"type": "json_object"} if format == "json" else None
        )
        return {
            "message": {
                "content": completion.choices[0].message.content,
                "role": "assistant"
            }
        }
    
    def prepare_prompt(self, system_prompt: str, user_query: str, context: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_query})
        return messages