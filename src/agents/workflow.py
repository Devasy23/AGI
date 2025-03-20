import streamlit as st
from typing import Dict, List, Any
from langgraph.graph import StateGraph, END

from .models import AgentRes, State
from src.llm.llm_factory import LLMFactory
from src.tools import ToolFactory
from src.memory import SimpleMemory
from src.utils.ui_helper import StreamlitUI

class AgentWorkflow:
    def __init__(self):
        self.llm = LLMFactory.create_llm()
        self.memory = SimpleMemory()
        self.tools = ToolFactory.get_tools()
        self.ui = StreamlitUI()
    
    def save_memory(self, lst_res: List[AgentRes], user_q: str) -> List[Dict[str, str]]:
        # Add to memory and get context
        self.memory.add_memory(lst_res, user_q)
        return self.memory.get_relevant_context(user_q)

    def node_agent(self, state: State) -> Dict[str, List[AgentRes]]:
        self.ui.update_current_step("Agent thinking...")
        str_tools = "\n".join([f"{i+1}. `{tool.name}`: {tool.description}" 
                              for i, tool in enumerate(self.tools.values())])
        prompt_tools = f"You can use the following tools:\n{str_tools}"
        
        # Get context from memory
        context = state["chat_history"] + self.save_memory(state["lst_res"], state["user_q"])
        
        messages = self.llm.prepare_prompt(
            system_prompt=self.get_agent_prompt() + "\n" + prompt_tools,
            user_query=state["user_q"],
            context=context
        )
        
        llm_res = self.llm.chat(messages=messages, format="json")
        agent_res = AgentRes.from_llm(llm_res)
        return {"lst_res":[agent_res]}

    def node_agent_2(self, state: State) -> Dict[str, List[AgentRes]]:
        self.ui.update_current_step("Second agent thinking...")
        str_tools = "\n".join([f"{i+1}. `{tool.name}`: {tool.description}" 
                              for i, tool in enumerate(self.tools.values())])
        prompt_tools = f"You can use the following tools:\n{str_tools}"
        
        output_text = state["output"].get("tool_output", "") if isinstance(state["output"], dict) else state["output"].tool_output
        
        # Get context from memory
        context = state["chat_history"] + self.save_memory(state["lst_res"], state["user_q"])
        
        messages = self.llm.prepare_prompt(
            system_prompt=self.get_agent_2_prompt() + "\n" + prompt_tools,
            user_query=output_text,
            context=context
        )
        
        llm_res = self.llm.chat(messages=messages, format="json")
        agent_res = AgentRes.from_llm(llm_res)
        return {"lst_res":[agent_res]}

    def node_tool(self, state: State) -> Dict:
        res = state["lst_res"][-1]
        self.ui.update_current_step(f"Using {res.tool_name}...")
        
        tool = self.tools[res.tool_name]
        agent_res = AgentRes(
            tool_name=res.tool_name,
            tool_input=res.tool_input,
            tool_output=str(tool.invoke(**res.tool_input))
        )
        
        return {"output": agent_res} if res.tool_name == "final_answer" else {"lst_res": [agent_res]}

    def human_edges(self, state: State) -> str:
        self.ui.update_current_step("Human decision point...")
        choice = self.ui.create_human_feedback_buttons()
        return choice if choice else "Human"

    def conditional_edges(self, state: State) -> str:
        last_res = state["lst_res"][-1]
        next_node = last_res.tool_name if isinstance(state["lst_res"], list) else "final_answer"
        self.ui.update_current_step(f"Moving to {next_node}...")
        return next_node

    def human_node(self, state: State) -> State:
        self.ui.update_current_step("Waiting for human input...")
        return state

    def create_graph(self) -> StateGraph:
        workflow = StateGraph(State)
        
        # Agent 1
        workflow.add_node("Agent1", action=self.node_agent)
        workflow.set_entry_point("Agent1")
        workflow.add_node("tool_browser", action=self.node_tool)
        workflow.add_node("final_answer", action=self.node_tool)
        workflow.add_edge(start_key="tool_browser", end_key="Agent1")
        workflow.add_conditional_edges(source="Agent1", path=self.conditional_edges)
        
        # Human
        workflow.add_node("Human", action=self.human_node)
        workflow.add_conditional_edges(source="final_answer", path=self.human_edges)
        
        # Agent 2
        workflow.add_node("Agent2", action=self.node_agent_2)
        workflow.add_node("tool_wikipedia", action=self.node_tool)
        workflow.add_edge(start_key="tool_wikipedia", end_key="Agent2")
        workflow.add_conditional_edges(source="Agent2", path=self.conditional_edges)
        
        return workflow.compile()

    @staticmethod
    def get_agent_prompt() -> str:
        return """
        You know everything, you must answer every question from the user, you can use the list of tools provided to you.
        Your goal is to provide the user with the best possible answer, including key information about the sources and tools used.

        Note, when using a tool, you provide the tool name and the arguments to use in JSON format. 
        For each call, you MUST ONLY use one tool AND the response format must ALWAYS be in the pattern:
        ```json
        {"name":"<tool_name>", "parameters": {"<tool_input_key>":<tool_input_value>}}
        ```
        Remember, do NOT use any tool with the same query more than once.
        Remember, if the user doesn't ask a specific question, you MUST use the `final_answer` tool directly.
        Remember, parameters are case-sensitive, and must be written exactly as in the tool description.

        Every time the user asks a question, you take note in the memory.
        Every time you find some information related to the user's question, you take note in the memory.

        You should aim to collect information from a diverse range of sources before providing the answer to the user. 
        Once you have collected plenty of information to answer the user's question use the `final_answer` tool.
        """

    @staticmethod
    def get_agent_2_prompt() -> str:
        return """
        Your goal is to use the `tool_wikipedia` ONLY ONCE to enrich the information already available.
        Note, when using a tool, you provide the tool name and the arguments to use in JSON format. 
        For each call, you MUST ONLY use one tool AND the response format must ALWAYS be in the json response pattern given below:
        
        {"name":"<tool_name>", "parameters": {"<tool_input_key>":<tool_input_value>}}
        
        First you must use the `tool_wikipedia`, then elaborate the information to answer the user's question with `final_answer` tool.
        """