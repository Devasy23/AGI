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
        self.conversation_window = 50  # Increased window size for better context
    
    def add(self, message):
        self.messages.append(message)
        # Only trim if significantly over the window size to avoid frequent trimming
        if len(self.messages) > self.conversation_window + 10:
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
            # Add user query to conversation context
            self.add({"role": "user", "content": user_q})
            
        for res in [res for res in lst_res if res.tool_output is not None]:
            memory_entry = {
                "query": user_q,
                "tool": res.tool_name,
                "input": json.dumps(res.tool_input),
                "output": res.tool_output,
                "timestamp": datetime.now().isoformat()
            }
            self.memories.append(memory_entry)
            
            # Add tool interactions and responses to conversation context
            self.add({
                "role": "assistant",
                "content": f"Using {res.tool_name} to help answer your question."
            })
            self.add({
                "role": "system",
                "content": f"Result: {res.tool_output}"
            })
    
    def get_relevant_context(self, query: str) -> List[Dict[str, str]]: 
        context = []
        
        # Add more recent conversation history
        context.extend(self.messages[-15:])  # Last 15 messages for immediate context
        
        # Add relevant tool interactions with better relevance matching
        relevant_memories = []
        query_words = set(query.lower().split())
        
        for memory in self.memories:
            memory_text = f"{memory['query']} {memory['output']}".lower()
            # Check for any word overlap between query and memory
            if any(word in memory_text for word in query_words):
                relevant_memories.append(memory)
        
        # Sort by timestamp and take most recent relevant memories
        relevant_memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        for memory in relevant_memories[:5]:  # Include more relevant memories
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
        """Save memory state to a file"""
        state = {
            "messages": self.messages,
            "memories": self.memories,
            "queries": self.queries
        }
        try:
            with open("memory_state.json", "w") as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Failed to save memory state: {e}")
    
    def load_state(self) -> None:
        """Load memory state from a file"""
        try:
            with open("memory_state.json", "r") as f:
                state = json.load(f)
                self.messages = state.get("messages", [])
                self.memories = state.get("memories", [])
                self.queries = state.get("queries", [])
        except FileNotFoundError:
            # No previous state exists
            pass
        except Exception as e:
            print(f"Failed to load memory state: {e}")