from abc import ABC, abstractmethod
from typing import Any, Dict, Callable
from pydantic import BaseModel
from langchain.tools import BaseTool as LangChainBaseTool

class BaseTool(LangChainBaseTool):
    """Base class for all tools in the system."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def invoke(self, **kwargs) -> Any:
        """Execute the tool with the given parameters"""
        return self._run(**kwargs)
    
    def to_dict(self) -> Dict[str, str]:
        """Return tool info as dictionary"""
        return {
            "name": self.name,
            "description": self.description
        }