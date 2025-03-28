from crewai import Agent
from src.llm.crew_llm import create_llm
from src.tools.crew_tools import get_search_tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool
)

class CrewAgentFactory:
    @staticmethod
    def create_research_agent():
        return Agent(
            role='Research Agent',
            goal='Find accurate and up-to-date information from multiple sources',
            backstory='''Expert at searching and gathering information from multiple sources.
                        Skilled at formulating effective search queries and extracting key information.
                        Always verifies information from multiple sources when possible.''',
            allow_delegation=False,
            llm=create_llm(),
            tools=get_search_tools(),  # Use our Langchain-based search tools
            verbose=True
        )

    @staticmethod
    def create_file_expert():
        return Agent(
            role='File Expert',
            goal='Read and analyze files and directories efficiently',
            backstory='''Specializes in reading and processing file contents.
                        Expert at extracting relevant information from various file types.
                        Focuses on organizing and presenting file data clearly.''',
            allow_delegation=False,
            llm=create_llm(),
            tools=[
                DirectoryReadTool(),
                FileReadTool()
            ],
            verbose=True
        )

    @staticmethod
    def create_synthesizer_agent():
        return Agent(
            role='Information Synthesizer',
            goal='Combine and present information in a clear, comprehensive, and well-structured way',
            backstory='''Expert at analyzing and synthesizing information from multiple sources.
                        Skilled at identifying key points and creating coherent narratives.
                        Ensures all information is properly attributed and organized.
                        Highlights any contradictions or uncertainties in the information.''',
            allow_delegation=False,
            llm=create_llm(),
            tools=[FileReadTool()],  # Synthesizer mainly needs to read and combine information
            verbose=True
        )