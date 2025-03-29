from crewai import Agent
from src.llm.crew_llm import create_llm
from src.tools.crew_tools import get_search_tools

class CrewAgentFactory:
    @staticmethod
    def create_planner_agent():
        """Creates a planner agent that determines query type and best approach"""
        return Agent(
            role='Query Planner',
            goal='Analyze user queries and determine the best approach for responding',
            backstory='''Expert at understanding user intent and efficiently routing queries.
                        Identifies whether a query needs external information or just a simple response.
                        Can determine if a query is a basic greeting, question, or complex research need.
                        Optimizes the agent workflow to avoid unnecessary tool usage.''',
            allow_delegation=True,
            llm=create_llm(),
            tools=[],  # Planner doesn't need tools, just decision-making capability
            verbose=True
        )
        
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
            tools=[],  # Synthesizer doesn't need tools, just synthesis capability
            verbose=True
        )