from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools.base import ToolException
from typing import Optional, Any, List
from pydantic import Field

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