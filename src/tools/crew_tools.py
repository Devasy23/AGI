from typing import Optional
from pydantic import BaseModel, Field
from crewai.tools import CrewStructuredTool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Input schemas for tools
class SearchQuery(BaseModel):
    """Input schema for search tools"""
    query: str = Field(..., description="The search query to execute")

class AnswerInput(BaseModel):
    """Input schema for answer formulation"""
    text: str = Field(..., description="The text to process into a final answer")

def browser_search(query: str) -> str:
    """Execute a browser search using DuckDuckGo"""
    search = DuckDuckGoSearchRun()
    return search.run(query)

def wikipedia_search(query: str) -> str:
    """Execute a Wikipedia search"""
    search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return search.run(query)

def formulate_answer(text: str) -> str:
    """Process and return the final answer"""
    return text

# Create structured tools
BrowserSearchTool = CrewStructuredTool.from_function(
    name="browser_search",
    description="Search on DuckDuckGo browser. Useful for finding current information from the web.",
    args_schema=SearchQuery,
    func=browser_search
)

WikipediaSearchTool = CrewStructuredTool.from_function(
    name="wikipedia_search",
    description="Search on Wikipedia. Use this for finding detailed, factual information from Wikipedia articles.",
    args_schema=SearchQuery,
    func=wikipedia_search
)

AnswerFormulationTool = CrewStructuredTool.from_function(
    name="formulate_answer",
    description="Formulate a comprehensive answer using all available information.",
    args_schema=AnswerInput,
    func=formulate_answer
)

# Optional: Add caching for efficiency
def cache_func(args, result):
    """Determine if result should be cached"""
    # Cache based on input length and result size
    return len(str(args)) < 1000 and len(str(result)) < 10000

BrowserSearchTool.cache_function = cache_func
WikipediaSearchTool.cache_function = cache_func