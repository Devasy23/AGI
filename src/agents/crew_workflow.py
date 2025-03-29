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
        # Initialize all agents
        planner = self.agent_factory.create_planner_agent()
        researcher = self.agent_factory.create_research_agent()
        synthesizer = self.agent_factory.create_synthesizer_agent()
        
        # Create context string from chat history
        context_str = "\n".join([msg["content"] for msg in chat_history])
        
        # First, create a planning task to determine how to handle the query
        planning_task = Task(
            description=f"""Analyze this user query and determine the best approach:
Query: "{query}"

Context from previous interactions:
{context_str}

Your job is to determine:
1. If this is a simple greeting or conversation (like "hello", "hi", "how are you")
2. If this requires searching the internet for information

Return ONLY ONE of these exact responses:
- "SIMPLE_RESPONSE" - For greetings and basic conversation
- "INTERNET_SEARCH" - When we need to search online for information
""",
            agent=planner,
            expected_output="A single decision about how to handle the query"
        )
        
        # Run the planning task independently
        planning_crew = Crew(
            agents=[planner],
            tasks=[planning_task],
            verbose=True,
            process="sequential"
        )
        
        planning_result = planning_crew.kickoff()
        planning_decision = str(planning_result).strip() if hasattr(planning_result, 'raw') else str(planning_result).strip()
        
        # Log the planning decision
        self.ui.add_chat_message("system", f"Planning decision: {planning_decision}", is_progress=True)
        
        # Different workflows based on the planning decision
        if planning_decision == "SIMPLE_RESPONSE":
            # For basic conversations, use only the synthesizer
            simple_task = Task(
                description=f"""Respond to this simple greeting or conversation:
Query: "{query}"

Context from previous interactions:
{context_str}

Provide a friendly, conversational response without using any search tools.
""",
                agent=synthesizer,
                expected_output="A friendly conversational response"
            )
            
            crew = Crew(
                agents=[synthesizer],
                tasks=[simple_task],
                verbose=True,
                process="sequential"
            )
            
        else:  # "INTERNET_SEARCH" or any other response
            # For internet searches
            research_task = Task(
                description=f"""Research the following query:
Query: "{query}"

Context from previous interactions:
{context_str}

Focus on finding information from online sources.
""",
                agent=researcher,
                expected_output="A detailed analysis with information from online sources"
            )
            
            synthesis_task = Task(
                description="Synthesize findings into a comprehensive response",
                agent=synthesizer,
                expected_output="A clear, well-structured response",
                context=[research_task]
            )
            
            crew = Crew(
                agents=[researcher, synthesizer],
                tasks=[research_task, synthesis_task],
                verbose=True,
                process="sequential"
            )
            
        # Execute the chosen workflow and get result
        result = crew.kickoff()
        
        # Convert CrewOutput to string
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