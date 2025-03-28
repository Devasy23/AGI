# Import crewAI's native tools
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper
)

class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Search the internet using DuckDuckGo. Use this for general queries and finding current information."
    search: DuckDuckGoSearchAPIWrapper = Field(default_factory=DuckDuckGoSearchAPIWrapper)

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error performing DuckDuckGo search: {str(e)}"

class WikipediaSearchTool(BaseTool):
    name: str = "Wikipedia Research"
    description: str = "Search Wikipedia for factual information and detailed explanations."
    search: WikipediaAPIWrapper = Field(default_factory=WikipediaAPIWrapper)

    def _run(self, query: str) -> str:
        """Search Wikipedia and return results"""
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

def get_search_tools():
    """Get available search tools"""
    return [DuckDuckGoSearchTool(), WikipediaSearchTool()]

__all__ = [
    'DuckDuckGoSearchTool',
    'WikipediaSearchTool',
    'get_search_tools'
]