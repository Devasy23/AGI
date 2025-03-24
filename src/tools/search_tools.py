from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from .base_tool import BaseTool
from src.llm.mcp_client import MCPClient

class BrowserSearchTool(BaseTool):
    name = "tool_browser"
    description = "Search on DuckDuckGo browser by passing the input `query`"
    
    def __init__(self):
        self._search = DuckDuckGoSearchRun()
    
    def invoke(self, query: str) -> str:
        return self._search.run(query)

class WikipediaSearchTool(BaseTool):
    name = "tool_wikipedia"
    description = "Search on Wikipedia by passing the input `query`. The input `query` must be short keywords, not a long text"
    
    def __init__(self):
        self._search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    def invoke(self, query: str) -> str:
        return self._search.run(query)

class FinalAnswerTool(BaseTool):
    name = "final_answer"
    description = "Returns a natural language response to the user by passing the input `text`. You should provide as much context as possible and specify the source of the information."
    
    def invoke(self, text: str) -> str:
        return text

class MCPTool(BaseTool):
    name = "tool_mcp"
    description = "Interact with Model Context Protocol (MCP) server to fetch or update context"
    
    def __init__(self):
        self.client = MCPClient()
    
    def invoke(self, action: str, data: dict = None) -> str:
        """
        Handle MCP operations
        :param action: The MCP action to perform (e.g., 'fetch', 'update')
        :param data: Optional data payload for the action
        :return: Result of the MCP operation
        """
        if action == "fetch":
            return self._fetch_context(data)
        elif action == "update":
            return self._update_context(data)
        else:
            raise ValueError(f"Unsupported MCP action: {action}")
    
    def _fetch_context(self, query_data: dict) -> str:
        try:
            result = self.client.fetch_context(query_data)
            return str(result)
        except Exception as e:
            return f"Error fetching context: {str(e)}"
    
    def _update_context(self, context_data: dict) -> str:
        try:
            result = self.client.update_context(context_data)
            return str(result)
        except Exception as e:
            return f"Error updating context: {str(e)}"