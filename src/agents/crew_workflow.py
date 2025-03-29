from typing import List, Dict
import streamlit as st
from crewai import Task, Crew
from src.utils.ui_helper import StreamlitUI
import os
import shutil
from .crew_agents import CrewAgentFactory
from .models import AgentRes
from src.knowledge import KnowledgeBase
from src.config.llm_config import LLMConfig

class CrewWorkflow:
    def __init__(self, memory=None):
        self.ui = StreamlitUI()
        self.agent_factory = CrewAgentFactory()
        self.knowledge_base = KnowledgeBase() if 'knowledge_base' not in st.session_state else st.session_state.knowledge_base
        
        # Copy PDF files to current working directory to fix CrewAI path issues
        self._prepare_knowledge_files()

    def _prepare_knowledge_files(self):
        """
        Copy PDF files from the knowledge directory to the current working directory
        to work around CrewAI's path handling issues
        """
        try:
            for source in self.knowledge_base.get_enabled_sources():
                if source.source_type == "pdf" and os.path.exists(source.content_path):
                    # Get just the filename
                    filename = os.path.basename(source.content_path)
                    # Copy to current working directory
                    target_path = os.path.join(os.getcwd(), filename)
                    if not os.path.exists(target_path):
                        shutil.copy2(source.content_path, target_path)
                        st.write(f"Prepared knowledge file: {filename}")
        except Exception as e:
            st.error(f"Error preparing knowledge files: {str(e)}")

    def process_query(self, query: str, chat_history: List[Dict[str, str]], lst_res: List) -> str:
        # Initialize all agents
        planner = self.agent_factory.create_planner_agent()
        researcher = self.agent_factory.create_research_agent()
        synthesizer = self.agent_factory.create_synthesizer_agent()
        
        # Create context string from chat history
        context_str = "\n".join([msg["content"] for msg in chat_history])
        
        # Get knowledge sources for CrewAI
        knowledge_sources = self.knowledge_base.get_crew_knowledge_sources()
        if knowledge_sources:
            st.write("🔍 Vector store available for queries with", len(knowledge_sources), "sources")
            
        # First, create a planning task to determine how to handle the query
        planning_task = Task(
            description=f"""Analyze this user query and determine the best approach:
Query: "{query}"

Context from previous interactions:
{context_str}

Your job is to determine:
1. If this is a simple greeting or conversation (like "hello", "hi", "how are you")
2. If this requires searching the internet for information
3. If this can be answered using our knowledge base

Return ONLY ONE of these exact responses:
- "SIMPLE_RESPONSE" - For greetings and basic conversation
- "INTERNET_SEARCH" - When we need to search online for information
- "KNOWLEDGE_BASE" - When we can use our knowledge base to answer the query
""",
            agent=planner,
            expected_output="A single decision about how to handle the query"
        )
        
        # Run the planning task independently
        planning_crew = Crew(
            agents=[planner],
            tasks=[planning_task],
            verbose=True,
            process="sequential",
            knowledge_sources=knowledge_sources,
            embedder={
                "provider": "google",
                "config": {
                    "model": "models/text-embedding-004",
                    "api_key": LLMConfig().get_gemini_api_key(),
                }
            }
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
        
        elif planning_decision == "KNOWLEDGE_BASE" and knowledge_sources:
            # For knowledge base queries
            st.write("🔍 Querying vector store for:", query)
            knowledge_task = Task(
                description=f"""Answer the following query using our knowledge base:
Query: "{query}"

Context from previous interactions:
{context_str}

Use only the knowledge base to answer the query. Be specific about which sources you're using.
""",
                agent=synthesizer,
                expected_output="A detailed answer based on knowledge base content"
            )
            
            # Create crew with knowledge sources
            crew = Crew(
                agents=[synthesizer],
                tasks=[knowledge_task],
                verbose=True,
                process="sequential",
                knowledge_sources=knowledge_sources,
                embedder={
                    "provider": "google",
                    "config": {
                        "model": "models/text-embedding-004",
                        "api_key": LLMConfig().get_gemini_api_key(),
                    }
                }
            )
            
            # Log knowledge sources being used
            source_names = [source.name for source in self.knowledge_base.get_enabled_sources()]
            self.ui.add_chat_message("system", f"📚 Using knowledge sources: {', '.join(source_names)}", is_progress=True)
            
        else:  # "INTERNET_SEARCH" or if no knowledge sources available
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
                context=[research_task],
                embedder={
                    "provider": "google",
                    "config": {
                        "model": "models/text-embedding-004",
                        "api_key": LLMConfig().get_gemini_api_key(),
                    }
                }
            )
            
            # Create crew with knowledge sources if available (to enhance research)
            crew_kwargs = {
                "agents": [researcher, synthesizer],
                "tasks": [research_task, synthesis_task],
                "verbose": True,
                "process": "sequential",
                "knowledge_sources": knowledge_sources if knowledge_sources else None,
                "embedder": {
                    "provider": "google",
                    "config": {
                        "model": "models/text-embedding-004",
                        "api_key": LLMConfig().get_gemini_api_key(),
                    }
                }
            }
            print (f"Knowledge sources: {knowledge_sources}")
            # Add knowledge sources if available
            if knowledge_sources:
                crew_kwargs["knowledge_sources"] = knowledge_sources
                source_names = [source.name for source in self.knowledge_base.get_enabled_sources()]
                self.ui.add_chat_message("system", f"Using knowledge sources as supplementary context: {', '.join(source_names)}", is_progress=True)
            
            crew = Crew(**crew_kwargs)
            
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