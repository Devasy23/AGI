from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools.base import ToolException
from typing import Optional, Any, List
from pydantic import Field
from src.llm.mcp_client import MCPClient

class BrowserSearchTool(BaseTool):
    name: str = Field(default="tool_browser")
    description: str = Field(default="Search on DuckDuckGo browser by passing the input `query`")
    
    def __init__(self) -> None:
        super().__init__()
        self._search = DuckDuckGoSearchRun()
    
    def _run(self, query: str) -> str:
        return self._search.run(query)

    async def _arun(self, query: str) -> str:
        return self._run(query)

class WikipediaSearchTool(BaseTool):
    name: str = Field(default="tool_wikipedia")
    description: str = Field(default="Search on Wikipedia by passing the input `query`. The input `query` must be short keywords, not a long text")
    
    def __init__(self) -> None:
        super().__init__()
        self._search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    def _run(self, query: str) -> str:
        return self._search.run(query)

    async def _arun(self, query: str) -> str:
        return self._run(query)

class FinalAnswerTool(BaseTool):
    name: str = Field(default="final_answer")
    description: str = Field(default="Returns a natural language response to the user by passing the input `text`. You should provide as much context as possible and specify the source of the information.")
    
    def _run(self, text: str) -> str:
        return text

    async def _arun(self, text: str) -> str:
        return self._run(text)

class MCPTool(BaseTool):
    name: str = Field(default="tool_mcp")
    description: str = Field(default="Interact with Model Context Protocol (MCP) server to fetch or update context")
    
    def __init__(self) -> None:
        super().__init__()
        self.client = MCPClient()
    
    def _run(self, action: str, data: Optional[dict] = None) -> str:
        if action == "fetch":
            return self._fetch_context(data)
        elif action == "update":
            return self._update_context(data)
        else:
            raise ToolException(f"Unsupported action: {action}")

    async def _arun(self, action: str, data: Optional[dict] = None) -> str:
        return self._run(action, data)

    def _fetch_context(self, query_data: dict) -> str:
        try:
            result = self.client.fetch_context(query_data)
            return str(result)
        except Exception as e:
            raise ToolException(f"Error fetching context: {str(e)}")

    def _update_context(self, context_data: dict) -> str:
        try:
            result = self.client.update_context(context_data)
            return str(result)
        except Exception as e:
            raise ToolException(f"Error updating context: {str(e)}")