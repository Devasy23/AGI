from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from .base_tool import BaseTool

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