from typing import Dict, List, Any
from crewai import Agent, Task, Crew, Process
from src.llm.llm_factory import LLMFactory
from src.tools import ToolFactory
from src.memory import SimpleMemory
from src.utils.ui_helper import StreamlitUI
from .models import AgentRes, State

class CrewWorkflow:
    def __init__(self):
        self.crew_llm = LLMFactory.create_crew_llm()
        self.memory = SimpleMemory()
        self.tools = ToolFactory.get_tools()
        self.ui = StreamlitUI()

    def create_agents(self):
        # Get tools directly since they are now LangChain tools
        browser_tool = self.tools["tool_browser"]
        wiki_tool = self.tools["tool_wikipedia"]
        answer_tool = self.tools["final_answer"]

        # Research Agent (Agent1 in original workflow)
        researcher = Agent(
            name="Researcher",
            goal="Find relevant information from web sources",
            backstory="You are an expert researcher with access to web search tools.",
            llm=self.crew_llm,
            tools=[browser_tool],
            verbose=True
        )

        # Analyst Agent (Agent2 in original workflow)
        analyst = Agent(
            name="Analyst",
            goal="Enrich and verify information using Wikipedia",
            backstory="You are an expert analyst who verifies and enriches information.",
            llm=self.crew_llm,
            tools=[wiki_tool],
            verbose=True
        )

        # Synthesizer Agent (final_answer handler)
        synthesizer = Agent(
            name="Synthesizer",
            goal="Synthesize information into clear, complete answers",
            backstory="You are an expert at combining information into clear answers.",
            llm=self.crew_llm,
            tools=[answer_tool],
            verbose=True
        )

        return researcher, analyst, synthesizer

    def create_tasks(self, query: str, researcher, analyst, synthesizer):
        # Add progress update for task creation
        self.ui.add_chat_message("system", "Creating research tasks...", is_progress=True)
        
        research_task = Task(
            description=f"Research information about: {query}",
            agent=researcher,
            context="Use web search to find relevant information.",
            expected_output="Detailed research findings"
        )

        analysis_task = Task(
            description="Verify and enrich the research findings",
            agent=analyst,
            context="Use Wikipedia to verify and add depth to the research.",
            expected_output="Verified and enriched information",
            dependencies=[research_task]
        )

        synthesis_task = Task(
            description="Create a comprehensive answer",
            agent=synthesizer,
            context="Combine all information into a clear, complete response.",
            expected_output="Final comprehensive answer",
            dependencies=[analysis_task]
        )

        return [research_task, analysis_task, synthesis_task]

    async def run(self, query: str):
        # Create agents
        researcher, analyst, synthesizer = self.create_agents()
        
        # Create tasks
        tasks = self.create_tasks(query, researcher, analyst, synthesizer)
        
        # Create crew
        crew = Crew(
            agents=[researcher, analyst, synthesizer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        # Execute crew tasks with progress tracking
        self.ui.update_current_step("Starting research crew...")
        
        # Add task execution progress
        self.ui.add_chat_message("system", "Research crew starting work...", is_progress=True)
        
        try:
            result = await crew.kickoff()
            
            # Process results and update memory
            all_results = []
            for task in tasks:
                if hasattr(task, 'output'):
                    # Add task completion progress
                    self.ui.add_chat_message(
                        "system", 
                        f"Task completed: {task.description}", 
                        is_progress=True
                    )
                    
                    agent_res = AgentRes(
                        tool_name=task.agent.name.lower(),
                        tool_input={"query": query},
                        tool_output=task.output
                    )
                    all_results.append(agent_res)
                    self.memory.add_memory([agent_res], query)
            
            return result
            
        except Exception as e:
            self.ui.add_chat_message("system", f"Error: {str(e)}", is_progress=True)
            raise e