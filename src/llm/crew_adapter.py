from typing import Any, List, Dict, Optional
from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from .llm_interface import LLMInterface

class CrewLLMAdapter(BaseChatModel):
    """Adapter to make our LLM interface work with CrewAI"""
    
    def __init__(self, llm: LLMInterface):
        super().__init__()
        self.llm = llm
        
    def _convert_to_chat_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to our chat format"""
        converted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted_messages.append({"role": "assistant", "content": msg.content})
        return converted_messages
        
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> str:
        """Convert CrewAI messages to our format and handle response"""
        try:
            converted_messages = self._convert_to_chat_messages(messages)
            response = self.llm.chat(messages=converted_messages)
            
            if not response or "message" not in response:
                raise ValueError("Invalid response format from LLM")
                
            return response["message"]["content"]
        except Exception as e:
            # Log error and return a fallback response
            print(f"Error in CrewLLMAdapter: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try again."

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> str:
        """Async version of _generate"""
        return self._generate(messages, stop)

    @property
    def _llm_type(self) -> str:
        return "custom_llm"