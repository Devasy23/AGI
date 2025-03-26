from typing import Dict, Type
from .crew_tools import BrowserSearchTool, WikipediaSearchTool, AnswerFormulationTool

# Since we're using CrewAI's native tools, we don't need the factory pattern anymore
# Tools are instantiated directly where needed

__all__ = ['BrowserSearchTool', 'WikipediaSearchTool', 'AnswerFormulationTool']