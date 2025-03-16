from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel

class BaseTool(ABC):
    name: str
    description: str
    
    @abstractmethod
    def invoke(self, **kwargs) -> Any:
        """Execute the tool's functionality"""
        pass
    
    def to_dict(self) -> Dict[str, str]:
        """Return tool info as dictionary"""
        return {
            "name": self.name,
            "description": self.description
        }