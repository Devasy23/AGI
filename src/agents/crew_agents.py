from crewai import Agent
from src.llm.crew_llm import create_llm
from src.tools.crew_tools import BrowserSearchTool, WikipediaSearchTool, AnswerFormulationTool

class CrewAgentFactory:
    @staticmethod
    def create_research_agent():
        return Agent(
            role='Research Agent',
            goal='Find accurate and up-to-date information from online sources',
            backstory='''Expert at searching and gathering information from multiple sources.
                        Skilled at formulating effective search queries and extracting key information.
                        Always verifies information from multiple sources when possible.''',
            allow_delegation=False,
            llm=create_llm(),
            tools=[BrowserSearchTool()],
            verbose=True
        )

    @staticmethod
    def create_wikipedia_agent():
        return Agent(
            role='Wikipedia Expert',
            goal='Enhance information with detailed Wikipedia knowledge',
            backstory='''Specializes in finding and analyzing Wikipedia articles.
                        Expert at cross-referencing information and finding relevant articles.
                        Focuses on extracting factual, well-sourced information.''',
            allow_delegation=False,
            llm=create_llm(),
            tools=[WikipediaSearchTool()],
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
            tools=[AnswerFormulationTool()],
            verbose=True
        )