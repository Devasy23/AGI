import google.generativeai as genai
from typing import List, Dict, Any
from .llm_interface import LLMInterface
from ..config.llm_config import LLMConfig

class GeminiLLM(LLMInterface):
    def __init__(self):
        self.config = LLMConfig.get_config()
        genai.configure(api_key=self.config["gemini_api_key"])
        self.model = genai.GenerativeModel('gemini-pro')
    
    def chat(self, messages: List[Dict[str, str]], format: str = "json") -> Dict[str, Any]:
        chat = self.model.start_chat()
        if format == "json":
            # Add JSON format instruction to system prompt
            messages[0]["content"] += "\nYou must respond in valid JSON format."
        
        # Process messages in order
        for msg in messages:
            if msg["role"] != "assistant":  # Skip assistant messages as they're responses
                response = chat.send_message(msg["content"])
        
        return {
            "message": {
                "content": response.text,
                "role": "assistant"
            }
        }
    
    def prepare_prompt(self, system_prompt: str, user_query: str, context: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_query})
        return messages