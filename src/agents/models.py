import json
from pydantic import BaseModel
import typing
from typing import List, Dict, Optional

class AgentRes(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: Optional[str] = None
    
    @classmethod
    def from_llm(cls, res: dict):
        try:
            # Check if we have a valid response
            if not res.get("message", {}).get("content"):
                raise ValueError("Empty response from LLM")
                
            content = res["message"]["content"]
            # If content is empty JSON, use final_answer with an error message
            if content == '{}':
                return cls(
                    tool_name="final_answer",
                    tool_input={"text": "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."}
                )
            
            out = json.loads(content)
            # Validate required fields
            if "name" not in out or "parameters" not in out:
                raise ValueError(f"Invalid response format. Expected 'name' and 'parameters' fields but got: {out}")
                
            return cls(tool_name=out["name"], tool_input=out["parameters"])
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from LLM: {content}"
            raise ValueError(error_msg) from e
        except Exception as e:
            raise ValueError(f"Error processing LLM response: {str(e)}") from e

class State(typing.TypedDict):
    user_q: str
    chat_history: List[Dict[str, str]]
    lst_res: List[AgentRes]
    output: Dict