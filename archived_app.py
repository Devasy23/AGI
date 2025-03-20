import streamlit as st
import ollama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import typing
import json

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = None

# Create sidebar for current step
st.sidebar.markdown("### Current Step")
current_step_container = st.sidebar.empty()

def update_current_step(step):
    st.session_state.current_step = step
    current_step_container.info(step)

# Set up the LLM
llm = "gemma3:4b"

# Tool definitions
@tool("tool_browser")
def tool_browser(query:str) -> str:
    """Search on DuckDuckGo browser by passing the input `query`"""
    return DuckDuckGoSearchRun().run(query)

@tool("tool_wikipedia")
def tool_wikipedia(query:str) -> str:
    """Search on Wikipedia by passing the input `query`.
       The input `query` must be short keywords, not a long text"""
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run(query)

@tool("final_answer")
def final_answer(text:str) -> str:
    """Returns a natural language response to the user by passing the input `text`. 
    You should provide as much context as possible and specify the source of the information.
    """
    return text

dic_tools = {
    "tool_browser": tool_browser,
    "tool_wikipedia": tool_wikipedia,
    "final_answer": final_answer
}

# Agent response class
class AgentRes(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str | None = None
    
    @classmethod
    def from_llm(cls, res:dict):
        try:
            # Check if we have a valid response
            if not res.get("message", {}).get("content"):
                raise ValueError("Empty response from LLM")
                
            content = res["message"]["content"]
            # If content is empty JSON, use final_answer with an error message
            if content == '{}':
                return cls(
                    tool_name="final_answer",
                    tool_input={"text": "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."}
                )
            
            out = json.loads(content)
            # Validate required fields
            if "name" not in out or "parameters" not in out:
                raise ValueError(f"Invalid response format. Expected 'name' and 'parameters' fields but got: {out}")
                
            return cls(tool_name=out["name"], tool_input=out["parameters"])
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from LLM: {content}"
            st.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            st.error(f"Error from Ollama:\n{res}\n{str(e)}")
            raise e

# State class
class State(typing.TypedDict):
    user_q: str
    chat_history: list 
    lst_res: list[AgentRes]
    output: dict

# Agent prompts
prompt = """
You know everything, you must answer every question from the user, you can use the list of tools provided to you.
Your goal is to provide the user with the best possible answer, including key information about the sources and tools used.

Note, when using a tool, you provide the tool name and the arguments to use in JSON format. 
For each call, you MUST ONLY use one tool AND the response format must ALWAYS be in the pattern:
```json
{"name":"<tool_name>", "parameters": {"<tool_input_key>":<tool_input_value>}}
```
Remember, do NOT use any tool with the same query more than once.
Remember, if the user doesn't ask a specific question, you MUST use the `final_answer` tool directly.
Remember, parameters are case-sensitive, and must be written exactly as in the tool description. ie a dictionary

Every time the user asks a question, you take note in the memory.
Every time you find some information related to the user's question, you take note in the memory.

You should aim to collect information from a diverse range of sources before providing the answer to the user. 
Once you have collected plenty of information to answer the user's question use the `final_answer` tool.
"""

prompt_2 = """
Your goal is to use the `tool_wikipedia` ONLY ONCE to enrich the information already available.
Note, when using a tool, you provide the tool name and the arguments to use in JSON format. 
For each call, you MUST ONLY use one tool AND the response format must ALWAYS be in the pattern:
```json
{"name":"<tool_name>", "parameters": {"<tool_input_key>":<tool_input_value>}}
```
First you must use the `tool_wikipedia`, then elaborate the information to answer the user's question with `final_answer` tool.
"""

# Node functions
def node_agent(state):
    update_current_step("Agent thinking...")
    str_tools = "\n".join([str(n+1)+". `"+str(v.name)+"`: "+str(v.description) for n,v in enumerate(dic_tools.values())])
    prompt_tools = f"You can use the following tools:\n{str_tools}"
    
    messages = [{"role":"system", "content":prompt+"\n"+prompt_tools},
                *state["chat_history"],
                {"role":"user", "content":state["user_q"]},
                *save_memory(lst_res=state["lst_res"], user_q=state["user_q"])]
    
    llm_res = ollama.chat(model=llm, messages=messages, format="json")
    agent_res = AgentRes.from_llm(llm_res)
    return {"lst_res":[agent_res]}

def node_agent_2(state):
    update_current_step("Second agent thinking...")
    str_tools = "\n".join([str(n+1)+". `"+str(v.name)+"`: "+str(v.description) for n,v in enumerate(dic_tools.values())])
    prompt_tools = f"You can use the following tools:\n{str_tools}"
    
    # Access the output AgentRes object correctly
    output_text = state["output"].get("tool_output", "") if isinstance(state["output"], dict) else state["output"].tool_output
    
    messages = [{"role":"system", "content":prompt_2+"\n"+prompt_tools},
                *state["chat_history"],
                {"role":"user", "content":output_text},
                *save_memory(lst_res=state["lst_res"], user_q=state["user_q"])]
    
    llm_res = ollama.chat(model=llm, messages=messages, format="json")
    agent_res = AgentRes.from_llm(llm_res)
    return {"lst_res":[agent_res]}

def node_tool(state):
    update_current_step(f"Using {state['lst_res'][-1].tool_name}...")
    res = state["lst_res"][-1]
    
    tool_fn = dic_tools[res.tool_name]
    # Convert the input parameters to match the tool's expected format
    if res.tool_name in ["tool_browser", "tool_wikipedia"]:
        tool_args = {"query": res.tool_input.get("query", "")}
    else:
        tool_args = {"text": res.tool_input.get("text", "")}
    
    agent_res = AgentRes(
        tool_name=res.tool_name,
        tool_input=res.tool_input,
        tool_output=str(tool_fn.invoke(tool_args))
    )
    
    return {"output":agent_res} if res.tool_name == "final_answer" else {"lst_res":[agent_res]}

def human_node(state):
    update_current_step("Waiting for human input...")
    return state

# Edge functions
def conditional_edges(state):
    last_res = state["lst_res"][-1]
    next_node = last_res.tool_name if isinstance(state["lst_res"], list) else "final_answer"
    update_current_step(f"Moving to {next_node}...")
    return next_node

def human_edges(state):
    update_current_step("Human decision point...")
    
    # Create columns for the buttons
    col1, col2 = st.columns([1, 4])
    
    # Store the choice in session state to persist across reruns
    if 'human_choice' not in st.session_state:
        st.session_state.human_choice = None
    
    with col1:
        if st.button("Yes", key="yes_button"):
            st.session_state.human_choice = "Agent2"
    with col2:
        if st.button("No", key="no_button"):
            st.session_state.human_choice = "END"
    
    # Return the choice if made
    if st.session_state.human_choice:
        choice = st.session_state.human_choice
        st.session_state.human_choice = None  # Reset for next time
        return choice
    return "Human"

def save_memory(lst_res:list[AgentRes], user_q:str) -> list:
    memory = []
    for res in [res for res in lst_res if res.tool_output is not None]:
        memory.extend([
            {"role":"assistant", "content":json.dumps({"name":res.tool_name, "parameters":res.tool_input})},
            {"role":"user", "content":res.tool_output}
        ])
    
    if memory:
        memory += [{"role":"user", "content":(f"""
                This is just a reminder that my original query was `{user_q}`.
                Only answer to the original query, and nothing else, but use the information I gave you. 
                Provide as much information as possible when you use the `final_answer` tool.
                """)}]
    return memory

# Create the graph
def create_graph():
    workflow = StateGraph(State)
    
    # Agent 1
    workflow.add_node("Agent1", action=node_agent)
    workflow.set_entry_point("Agent1")
    workflow.add_node("tool_browser", action=node_tool)
    workflow.add_node("final_answer", action=node_tool)
    workflow.add_edge(start_key="tool_browser", end_key="Agent1")
    workflow.add_conditional_edges(source="Agent1", path=conditional_edges)
    
    # Human
    workflow.add_node("Human", action=human_node)
    workflow.add_conditional_edges(source="final_answer", path=human_edges)
    
    # Agent 2
    workflow.add_node("Agent2", action=node_agent_2)
    workflow.add_node("tool_wikipedia", action=node_tool)
    workflow.add_edge(start_key="tool_wikipedia", end_key="Agent2")
    workflow.add_conditional_edges(source="Agent2", path=conditional_edges)
    
    return workflow.compile()

# Streamlit UI
st.title("Multi-Agent Search Assistant")
st.write("Ask a question and the agents will search for information using multiple sources.")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask your question"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    # Create initial state
    initial_state = {
        'user_q': question,
        'chat_history': [],
        'lst_res': [],
        'output': {}
    }
    
    # Create and run graph
    graph = create_graph()
    
    # Create a progress container
    progress_container = st.empty()
    
    # Run the graph
    with st.spinner('Processing...'):
        result = graph.invoke(input=initial_state)
        final_answer = result['output'].tool_output
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        with st.chat_message("assistant"):
            st.write(final_answer)

# Show current step
if st.session_state.current_step:
    st.sidebar.markdown("### Current Step")
    st.sidebar.info(st.session_state.current_step)