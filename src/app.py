import streamlit as st
from agents.workflow import AgentWorkflow
from agents.models import State
from utils import StreamlitUI, EnvConfig
from memory import SimpleMemory

# Initialize UI and environment
ui = StreamlitUI()
ui.initialize_session_state()
current_step_container = ui.setup_sidebar()

# Setup environment configuration UI
EnvConfig.setup_env_ui()

# Initialize persistent memory in session state
if 'memory' not in st.session_state:
    st.session_state.memory = SimpleMemory()
    # Try to load previous state
    st.session_state.memory.load_state()

# Create main UI
st.title("Multi-Agent Search Assistant")
st.write("Ask a question and the agents will search for information using multiple sources.")

# Initialize workflow with existing memory
try:
    workflow = AgentWorkflow(memory=st.session_state.memory)
except ValueError as e:
    st.error(f"Configuration error: {str(e)}")
    st.stop()

# Display existing chat messages
ui.show_chat_messages()

# Handle user input
if question := st.chat_input("Ask your question"):
    # Add user message to chat
    ui.add_chat_message("user", question)
    
    # Create initial state
    initial_state = {
        'user_q': question,
        'chat_history': [],
        'lst_res': [],
        'output': {}
    }
    
    # Run the workflow
    try:
        with st.spinner('Processing...'):
            graph = workflow.create_graph()
            result = graph.invoke(initial_state)
            
            # Handle final answer
            if 'output' in result and hasattr(result['output'], 'tool_output'):
                final_answer = result['output'].tool_output
                ui.add_chat_message("assistant", final_answer)
            
            # Save memory state
            st.session_state.memory.save_state()
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e

# Update current step display
if st.session_state.current_step:
    current_step_container.info(st.session_state.current_step)