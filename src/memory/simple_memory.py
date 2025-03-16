from typing import List, Dict, Any
import json
from .memory_interface import MemoryInterface
from src.agents.models import AgentRes

class SimpleMemory(MemoryInterface):
    """A simple implementation of memory interface that stores messages in a list."""
    
    def __init__(self):
        self.messages = []
        self.memories: List[Dict[str, str]] = []
        self.queries: List[str] = []
    
    def add(self, message):
        self.messages.append(message)
        
    def get(self):
        return self.messages
        
    def clear(self):
        self.messages = []
    
    def add_memory(self, lst_res: List[AgentRes], user_q: str) -> None:
        if user_q not in self.queries:
            self.queries.append(user_q)
            
        for res in [res for res in lst_res if res.tool_output is not None]:
            memory_entry = {
                "query": user_q,
                "tool": res.tool_name,
                "input": json.dumps(res.tool_input),
                "output": res.tool_output
            }
            self.memories.append(memory_entry)
    
    def get_relevant_context(self, query: str) -> List[Dict[str, str]]: 
        # Simple implementation - return all memories as context
        # This can be enhanced with vector similarity search later
        context = []
        for memory in self.memories:
            context.extend([
                {"role": "assistant", 
                 "content": json.dumps({"name": memory["tool"], 
                                      "parameters": json.loads(memory["input"])})},
                {"role": "user", 
                 "content": memory["output"]}
            ])
        return context
    
    def save_state(self) -> None:
        # For now, just keep in memory
        # Can be enhanced to save to disk/database
        pass
    
    def load_state(self) -> None:
        # For now, just keep in memory
        # Can be enhanced to load from disk/database
        pass