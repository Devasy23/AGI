from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
from .base_tool import BaseTool

# Tool input schemas
class BrowserSearchInput(BaseModel):
    """Input for browser search tool"""
    query: str = Field(..., description="The search query to look up")

class WikipediaSearchInput(BaseModel):
    """Input for Wikipedia search tool"""
    query: str = Field(..., description="The Wikipedia search query")

class FinalAnswerInput(BaseModel):
    """Input for final answer tool"""
    text: str = Field(..., description="The final answer text")

class BrowserSearchTool(BaseTool):
    """Tool that adds the capability to search using DuckDuckGo."""
    name: str = "tool_browser"
    description: str = "Search on DuckDuckGo browser by passing the input `query`"
    args_schema: type[BrowserSearchInput] = BrowserSearchInput
    
    def __init__(self) -> None:
        super().__init__()
        self._search = DuckDuckGoSearchRun()
    
    def _run(self, query: str, **kwargs: Dict[str, Any]) -> str:
        """Execute the browser search."""
        return self._search.run(query)

class WikipediaSearchTool(BaseTool):
    """Tool that adds the capability to search using Wikipedia."""
    name: str = "tool_wikipedia"
    description: str = "Search on Wikipedia by passing the input `query`. The input `query` must be short keywords, not a long text"
    args_schema: type[WikipediaSearchInput] = WikipediaSearchInput
    
    def __init__(self) -> None:
        super().__init__()
        self._search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    def _run(self, query: str, **kwargs: Dict[str, Any]) -> str:
        """Execute the Wikipedia search."""
        return self._search.run(query)

class FinalAnswerTool(BaseTool):
    """Tool that provides a final answer to the user."""
    name: str = "final_answer"
    description: str = "Returns a natural language response to the user by passing the input `text`. You should provide as much context as possible and specify the source of the information."
    args_schema: type[FinalAnswerInput] = FinalAnswerInput
    
    def _run(self, text: str, **kwargs: Dict[str, Any]) -> str:
        """Return the final answer."""
        return text