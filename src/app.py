import streamlit as st
import asyncio
from agents.crew_workflow import CrewWorkflow
from utils import StreamlitUI, EnvConfig

# Initialize UI and environment
ui = StreamlitUI()
ui.initialize_session_state()
current_step_container = ui.setup_sidebar()

# Setup environment configuration UI
EnvConfig.setup_env_ui()

# Create main UI
st.title("Multi-Agent Search Assistant")
st.write("Ask a question and the agents will search for information using multiple sources.")

# Initialize workflow
try:
    workflow = CrewWorkflow()
except ValueError as e:
    st.error(f"Configuration error: {str(e)}")
    st.stop()

# Display existing chat messages
ui.show_chat_messages()

# Handle user input
if question := st.chat_input("Ask your question"):
    # Add user message to chat
    ui.add_chat_message("user", question)
    
    # Run the workflow
    try:
        with st.spinner('Processing...'):
            # Run the async workflow
            result = asyncio.run(workflow.run(question))
            
            # Add final answer to chat
            ui.add_chat_message("assistant", result)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Update current step display
if st.session_state.current_step:
    current_step_container.info(st.session_state.current_step)