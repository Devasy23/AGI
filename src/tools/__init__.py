from typing import Dict, Type
from .base_tool import BaseTool
from .search_tools import BrowserSearchTool, WikipediaSearchTool, FinalAnswerTool, MCPTool

class ToolFactory:
    _tools: Dict[str, Type[BaseTool]] = {
        "tool_browser": BrowserSearchTool,
        "tool_wikipedia": WikipediaSearchTool,
        "final_answer": FinalAnswerTool,
        "tool_mcp": MCPTool
    }
    
    @classmethod
    def get_tools(cls) -> Dict[str, BaseTool]:
        """Get instances of all available tools"""
        return {name: tool_cls() for name, tool_cls in cls._tools.items()}
    
    @classmethod
    def register_tool(cls, name: str, tool_cls: Type[BaseTool]) -> None:
        """Register a new tool"""
        cls._tools[name] = tool_cls

__all__ = ['BaseTool', 'BrowserSearchTool', 'WikipediaSearchTool', 'FinalAnswerTool', 'MCPTool', 'ToolFactory']