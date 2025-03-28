import streamlit as st
from agents.crew_workflow import CrewWorkflow
from agents.models import AgentRes
from utils.ui_helper import StreamlitUI
from utils.env_config import EnvConfig

# Initialize UI and environment
ui = StreamlitUI()
ui.initialize_session_state()
ui.setup_sidebar()

# Setup environment configuration UI
EnvConfig.setup_env_ui()

# Create main UI
st.title("Multi-Agent Search Assistant")
st.write("Ask a question and our crew of AI agents will work together to find the answer.")

# Initialize workflow without memory dependency
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
    
    try:
        # Process the query using CrewAI workflow
        result = workflow.process_query(
            query=question,
            chat_history=st.session_state.messages,
            lst_res=st.session_state.get('lst_res', [])
        )
        
        # Update chat with the result
        ui.add_chat_message("assistant", result)
        
        # Save new agent result
        if 'lst_res' not in st.session_state:
            st.session_state.lst_res = []
        st.session_state.lst_res.append(AgentRes(
            tool_name="final_answer",
            tool_input={"text": result},
            tool_output=result
        ))
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        ui.add_chat_message("assistant", "I apologize, but I encountered an error while processing your request.")
