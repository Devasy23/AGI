from typing import List, Union, Dict, Any, TypeVar, Generic, Protocol, Type, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from src.agents.models import AgentRes

class MemoryInterface(Protocol):
    """Interface for memory implementations."""
    
    def add(self, message: Union[Dict[str, Any], "AgentRes"]) -> None:
        """Add a message to memory."""
        ...
    
    def get(self) -> List[Union[Dict[str, Any], "AgentRes"]]:
        """Get all messages from memory."""
        ...
    
    def clear(self) -> None:
        """Clear all messages from memory."""
        ...