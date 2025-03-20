import google.generativeai as genai
from typing import List, Dict, Any
from .llm_interface import LLMInterface
from src.config.llm_config import LLMConfig
import json

class GeminiLLM(LLMInterface):
    def __init__(self):
        self.config = LLMConfig.get_config()
        genai.configure(api_key=self.config["gemini_api_key"])
        self.model = genai.GenerativeModel(self.config["model_name"])

    def chat(self, messages: List[Dict[str, str]], format: str = "json") -> Dict[str, Any]:
        chat = self.model.start_chat()
        if format == "json":
            # Add JSON format instruction to system prompt
            messages[0]["content"] += "\nResponse must be only valid JSON without any markdown formatting or code blocks."
        
        # Process messages in order
        for msg in messages:
            if msg["role"] != "assistant":  # Skip assistant messages as they're responses
                response = chat.send_message(msg["content"])
        
        response_text = response.text.strip()
        
        # Handle JSON response
        if format == "json":
            try:
                # First try to parse as is
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                # Try cleaning up markdown/code blocks if present
                cleaned_text = response_text
                if "```" in cleaned_text:
                    # Extract content between triple backticks if present
                    parts = cleaned_text.split("```")
                    for part in parts:
                        if part.strip() and not part.strip().startswith("json"):
                            cleaned_text = part.strip()
                            break
                
                try:
                    response_json = json.loads(cleaned_text)
                except json.JSONDecodeError:
                    # If still fails, return error response
                    return {
                        "message": {
                            "content": json.dumps({"error": "Failed to parse response", "raw_response": response_text}),
                            "role": "assistant"
                        }
                    }
                
            return {
                "message": {
                    "content": json.dumps(response_json),
                    "role": "assistant"
                }
            }
        
        # For non-JSON responses
        return {
            "message": {
                "content": response_text,
                "role": "assistant"
            }
        }

    def prepare_prompt(self, system_prompt: str, user_query: str, context: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_query})
        return messages