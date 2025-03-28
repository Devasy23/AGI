from typing import List, Dict
from crewai import Task, Crew
from src.utils.ui_helper import StreamlitUI
from .crew_agents import CrewAgentFactory
from .models import AgentRes

class CrewWorkflow:
    def __init__(self, memory=None):
        self.ui = StreamlitUI()
        self.agent_factory = CrewAgentFactory()
    
    def process_query(self, query: str, chat_history: List[Dict[str, str]], lst_res: List) -> str:
        # Initialize agents
        researcher = self.agent_factory.create_research_agent()
        file_expert = self.agent_factory.create_file_expert()
        synthesizer = self.agent_factory.create_synthesizer_agent()
        
        # Create context string from chat history
        context_str = "\n".join([msg["content"] for msg in chat_history])
        
        # Create tasks with explicit dependencies
        research_task = Task(
            description=f"Research the following query:\n{query}\n\nContext from previous interactions:\n{context_str}",
            agent=researcher,
            expected_output="A detailed analysis of the query with information from online sources"
        )
        
        file_analysis_task = Task(
            description="Analyze relevant files and documentation to find complementary information",
            agent=file_expert,
            expected_output="Relevant information found in files and documentation",
            context=[research_task]  # Use research results as context
        )
        
        synthesis_task = Task(
            description="Synthesize all findings into a comprehensive response",
            agent=synthesizer,
            expected_output="A clear, well-structured response that combines all gathered information",
            context=[research_task, file_analysis_task]  # Use both research and file analysis as context
        )
        
        # Create and run the crew with sequential execution
        crew = Crew(
            agents=[researcher, file_expert, synthesizer],
            tasks=[research_task, file_analysis_task, synthesis_task],
            verbose=True,
            process="sequential"  # Use sequential processing
        )
        
        # Execute tasks and get result
        result = crew.kickoff()
        
        # Convert CrewOutput to string (handle both CrewOutput object and string cases)
        if hasattr(result, 'raw'):
            result_str = str(result.raw)
        else:
            result_str = str(result)
        
        # Create a proper AgentRes object with string output
        final_result = AgentRes(
            tool_name="final_answer",
            tool_input={"text": result_str},
            tool_output=result_str
        )
        
        # Add to list of results
        if 'lst_res' not in globals():
            lst_res = []
        lst_res.append(final_result)
        
        return result_str