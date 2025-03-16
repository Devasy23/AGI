from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..agents.models import AgentRes

class MemoryInterface(ABC):
    @abstractmethod
    def add_memory(self, lst_res: List[AgentRes], user_q: str) -> None:
        """Add new memories from agent responses"""
        pass
    
    @abstractmethod
    def get_relevant_context(self, query: str) -> List[Dict[str, str]]:
        """Retrieve relevant context based on the query"""
        pass
    
    @abstractmethod
    def save_state(self) -> None:
        """Save the current memory state"""
        pass
    
    @abstractmethod
    def load_state(self) -> None:
        """Load the saved memory state"""
        pass