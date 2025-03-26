from typing import List, Dict
from crewai import Task, Crew
from src.utils.ui_helper import StreamlitUI
from src.memory import SimpleMemory
from .crew_agents import CrewAgentFactory

class CrewWorkflow:
    def __init__(self, memory=None):
        self.memory = memory if memory is not None else SimpleMemory()
        self.ui = StreamlitUI()
        self.agent_factory = CrewAgentFactory()
    
    def process_query(self, query: str, chat_history: List[Dict[str, str]], lst_res: List) -> str:
        # Initialize agents
        researcher = self.agent_factory.create_research_agent()
        wiki_expert = self.agent_factory.create_wikipedia_agent()
        synthesizer = self.agent_factory.create_synthesizer_agent()
        
        # Get context from memory
        context = self.get_memory_context(lst_res, query)
        context_str = "\n".join([msg["content"] for msg in chat_history + context])
        
        # Create tasks with explicit dependencies
        research_task = Task(
            description=f"Research the following query:\n{query}\n\nContext from previous interactions:\n{context_str}",
            agent=researcher
        )
        
        wiki_task = Task(
            description="Find relevant Wikipedia information to complement this research. Focus on adding factual, well-sourced information that enhances our understanding.",
            agent=wiki_expert,
            context=[research_task]  # Wiki task uses research results as context
        )
        
        synthesis_task = Task(
            description="Create a comprehensive answer by combining and analyzing all gathered information. Ensure proper attribution of sources and highlight any important insights or connections.",
            agent=synthesizer,
            context=[research_task, wiki_task]  # Synthesis uses both research and wiki results
        )
        
        # Create and run the crew with optimized task execution
        crew = Crew(
            agents=[researcher, wiki_expert, synthesizer],
            tasks=[research_task, wiki_task, synthesis_task],
            verbose=True,
            process=self.process_callback
        )
        
        self.ui.update_current_step("Crew working on your query...")
        result = crew.kickoff()
        
        # Update memory with the result
        if result:
            self.memory.add_interaction(query, result)
            
        return result
    
    def process_callback(self, task: Task) -> None:
        """Callback to update UI with task progress"""
        self.ui.update_current_step(f"Working on: {task.description[:50]}...")
    
    def get_memory_context(self, lst_res: List, query: str) -> List[Dict[str, str]]:
        """Get relevant context from memory"""
        return self.memory.get_relevant_context(query)