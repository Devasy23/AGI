from typing import List, Dict, Any, Union, TYPE_CHECKING
import json
from datetime import datetime
from .memory_interface import MemoryInterface

# Only import at type checking time
if TYPE_CHECKING:
    from src.agents.models import AgentRes

class SimpleMemory(MemoryInterface):
    """A simple implementation of memory interface that stores messages in a list."""
    
    def __init__(self):
        self.messages = []
        self.memories: List[Dict[str, str]] = []
        self.queries: List[str] = []
        self.conversation_window = 10  # Keep last N messages for context
    
    def add(self, message):
        self.messages.append(message)
        # Trim messages to keep only recent context
        if len(self.messages) > self.conversation_window:
            self.messages = self.messages[-self.conversation_window:]
        
    def get(self):
        return self.messages
        
    def clear(self):
        self.messages = []
        self.memories = []
        self.queries = []
    
    def add_memory(self, lst_res: List['AgentRes'], user_q: str) -> None:
        if user_q not in self.queries:
            self.queries.append(user_q)
            
        for res in [res for res in lst_res if res.tool_output is not None]:
            memory_entry = {
                "query": user_q,
                "tool": res.tool_name,
                "input": json.dumps(res.tool_input),
                "output": res.tool_output,
                "timestamp": datetime.now().isoformat()
            }
            self.memories.append(memory_entry)
            
            # Add tool interactions to conversation context
            self.add({
                "role": "system",
                "content": f"Tool {res.tool_name} was used with input: {res.tool_input}"
            })
            self.add({
                "role": "system",
                "content": f"Tool output: {res.tool_output}"
            })
    
    def get_relevant_context(self, query: str) -> List[Dict[str, str]]: 
        context = []
        
        # Add recent conversation history
        context.extend(self.messages[-5:])  # Last 5 messages for immediate context
        
        # Add relevant tool interactions
        relevant_memories = []
        for memory in self.memories:
            # Simple relevance check - can be enhanced with embeddings
            if (any(word in memory["query"].lower() for word in query.lower().split()) or
                any(word in memory["output"].lower() for word in query.lower().split())):
                relevant_memories.append(memory)
        
        # Sort by timestamp if available and take most recent
        relevant_memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        for memory in relevant_memories[:3]:  # Take top 3 relevant memories
            context.extend([
                {"role": "system", 
                 "content": f"Previously for query '{memory['query']}', tool {memory['tool']} was used:"},
                {"role": "assistant", 
                 "content": json.dumps({"name": memory["tool"], 
                                      "parameters": json.loads(memory["input"])})},
                {"role": "system", 
                 "content": f"Result: {memory['output']}"}
            ])
        
        return context
    
    def save_state(self) -> None:
        # TODO: Implement persistence
        pass
    
    def load_state(self) -> None:
        # TODO: Implement loading
        pass