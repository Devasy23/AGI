from abc import ABC, abstractmethod
from typing import Any, Dict, Callable
from pydantic import BaseModel
from langchain.tools import BaseTool as LangChainBaseTool

class BaseTool:
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function
    
    def invoke(self, **kwargs) -> Any:
        """Execute the tool with the given parameters"""
        return self.function(**kwargs)
    
    def to_dict(self) -> Dict[str, str]:
        """Return tool info as dictionary"""
        return {
            "name": self.name,
            "description": self.description
        }
    
    def to_langchain_tool(self) -> LangChainBaseTool:
        """Convert to LangChain tool format for CrewAI compatibility"""
        return LangChainBaseTool(
            name=self.name,
            description=self.description,
            func=lambda **kwargs: str(self.invoke(**kwargs))
        )