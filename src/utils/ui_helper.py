import streamlit as st
from typing import Optional, Any, Dict
from ..config import Config, MemoryConfig

class StreamlitUI:
    @staticmethod
    def initialize_session_state():
        """Initialize all required session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_step' not in st.session_state:
            st.session_state.current_step = None
        if 'human_choice' not in st.session_state:
            st.session_state.human_choice = None
        if 'env_vars' not in st.session_state:
            st.session_state.env_vars = Config.get_all()

    @staticmethod
    def setup_sidebar():
        """Setup the sidebar with current step indicator and configuration options"""
        st.sidebar.markdown("### Current Step")
        step_container = st.sidebar.empty()
        
        # Add memory configuration UI
        StreamlitUI.setup_memory_config_ui()
        
        return step_container

    @staticmethod
    def update_current_step(step: str, container: Optional[Any] = None):
        """Update the current step in session state and UI"""
        st.session_state.current_step = step
        if container:
            container.info(step)

    @staticmethod
    def show_chat_messages():
        """Display all chat messages"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    @staticmethod
    def add_chat_message(role: str, content: str):
        """Add a new message to chat history"""
        st.session_state.messages.append({"role": role, "content": content})
        with st.chat_message(role):
            st.write(content)

    @staticmethod
    def create_human_feedback_buttons():
        """Create human feedback buttons with proper layout"""
        col1, col2 = st.columns([1, 4])
        with col1:
            yes_clicked = st.button("Yes", key="yes_button")
            if yes_clicked:
                st.session_state.human_choice = "Agent2"
        with col2:
            no_clicked = st.button("No", key="no_button")
            if no_clicked:
                st.session_state.human_choice = "END"
        
        if st.session_state.human_choice:
            choice = st.session_state.human_choice
            st.session_state.human_choice = None
            return choice
        return None

    @staticmethod
    def setup_memory_config_ui():
        """Setup UI for memory configuration"""
        with st.sidebar.expander("Memory Settings"):
            # Vector Store Selection
            vector_store = st.selectbox(
                "Vector Store",
                options=["chroma", "qdrant", "faiss"],
                index=["chroma", "qdrant", "faiss"].index(MemoryConfig.vector_store)
            )
            
            # Embedding Model Selection
            embedding_model = st.selectbox(
                "Embedding Model",
                options=["sentence-transformers", "openai", "huggingface"],
                index=["sentence-transformers", "openai", "huggingface"].index(MemoryConfig.embedding_model)
            )
            
            # Model specific settings
            if embedding_model == "sentence-transformers":
                embedding_model_name = st.text_input(
                    "Model Name",
                    value=MemoryConfig.embedding_model_name
                )
            elif embedding_model == "openai":
                openai_key = st.text_input(
                    "OpenAI API Key",
                    value=MemoryConfig.openai_api_key,
                    type="password"
                )
            elif embedding_model == "huggingface":
                hf_key = st.text_input(
                    "HuggingFace API Key",
                    value=MemoryConfig.hf_api_key,
                    type="password"
                )
            
            # Vector store specific settings
            if vector_store == "chroma":
                persist_dir = st.text_input(
                    "Persist Directory",
                    value=MemoryConfig.chroma_persist_dir
                )
            elif vector_store == "qdrant":
                qdrant_url = st.text_input(
                    "Qdrant URL",
                    value=MemoryConfig.qdrant_url
                )
                qdrant_key = st.text_input(
                    "Qdrant API Key",
                    value=MemoryConfig.qdrant_api_key,
                    type="password"
                )
            
            if st.button("Save Memory Settings"):
                try:
                    # Update configuration
                    if "env_vars" not in st.session_state:
                        st.session_state.env_vars = {}
                    
                    st.session_state.env_vars.update({
                        "VECTOR_STORE": vector_store,
                        "EMBEDDING_MODEL": embedding_model,
                        "EMBEDDING_MODEL_NAME": embedding_model_name if embedding_model == "sentence-transformers" else MemoryConfig.embedding_model_name,
                        "OPENAI_API_KEY": openai_key if embedding_model == "openai" else MemoryConfig.openai_api_key,
                        "HF_API_KEY": hf_key if embedding_model == "huggingface" else MemoryConfig.hf_api_key,
                        "CHROMA_PERSIST_DIR": persist_dir if vector_store == "chroma" else MemoryConfig.chroma_persist_dir,
                        "QDRANT_URL": qdrant_url if vector_store == "qdrant" else MemoryConfig.qdrant_url,
                        "QDRANT_API_KEY": qdrant_key if vector_store == "qdrant" else MemoryConfig.qdrant_api_key
                    })
                    
                    # Validate new configuration
                    Config.validate_all()
                    st.success("Memory settings saved successfully!")
                except ValueError as e:
                    st.error(f"Invalid configuration: {str(e)}")