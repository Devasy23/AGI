import streamlit as st
from typing import Optional, Any, Dict
from src.config import Config, MemoryConfig

class StreamlitUI:
    @staticmethod
    def initialize_session_state():
        """Initialize all required session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_step' not in st.session_state:
            st.session_state.current_step = None
        if 'env_vars' not in st.session_state:
            st.session_state.env_vars = Config.get_all()
        if 'progress_updates' not in st.session_state:
            st.session_state.progress_updates = []
        if 'show_progress' not in st.session_state:
            st.session_state.show_progress = True

    @staticmethod
    def setup_sidebar():
        """Setup the sidebar with current step indicator and configuration options"""
        StreamlitUI.setup_memory_config_ui()

    @staticmethod
    def show_chat_messages():
        """Display all chat messages from the session state"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    @staticmethod
    def add_chat_message(role: str, content: str, is_progress: bool = False):
        """Add a message to the chat history and display it
        
        Args:
            role: The role of the message sender (user, assistant, system)
            content: The content of the message
            is_progress: Whether this is a progress update message
        """
        # Add to session state
        st.session_state.messages.append({"role": role, "content": content})
        
        # Display the message
        if is_progress:
            if st.session_state.show_progress:
                st.session_state.progress_updates.append({"role": role, "content": content})
                with st.expander("Progress Updates", expanded=True):
                    for update in st.session_state.progress_updates:
                        with st.chat_message(update["role"]):
                            st.markdown(update["content"])
        else:
            # Display immediately if it's not a progress message
            with st.chat_message(role):
                st.markdown(content)

    @staticmethod
    def setup_memory_config_ui():
        """Setup UI for memory configuration"""
        with st.sidebar.expander("Memory Settings"):
            # Vector Store Selection (Optional)
            current_vector_store = MemoryConfig.vector_store or "none"
            vector_store = st.selectbox(
                "Vector Store (Optional)",
                options=["none", "chroma", "qdrant", "faiss"],
                index=["none", "chroma", "qdrant", "faiss"].index(current_vector_store)
            )
            vector_store = None if vector_store == "none" else vector_store
            
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
            
            # Vector store specific settings (only show if a vector store is selected)
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
                    
                    # Update environment variables
                    env_updates = {
                        "VECTOR_STORE": vector_store if vector_store else "",
                        "EMBEDDING_MODEL": embedding_model,
                        "EMBEDDING_MODEL_NAME": embedding_model_name if embedding_model == "sentence-transformers" else MemoryConfig.embedding_model_name,
                        "OPENAI_API_KEY": openai_key if embedding_model == "openai" else MemoryConfig.openai_api_key,
                        "HF_API_KEY": hf_key if embedding_model == "huggingface" else MemoryConfig.hf_api_key,
                        "CHROMA_PERSIST_DIR": persist_dir if vector_store == "chroma" else MemoryConfig.chroma_persist_dir,
                        "QDRANT_URL": qdrant_url if vector_store == "qdrant" else MemoryConfig.qdrant_url,
                        "QDRANT_API_KEY": qdrant_key if vector_store == "qdrant" else MemoryConfig.qdrant_api_key
                    }
                    for key, value in env_updates.items():
                        if value:
                            st.session_state.env_vars[key] = value
                            st.session_state.env_vars[key] = MemoryConfig.clean_env_value(value)
                        else:
                            st.session_state.env_vars.pop(key, None)
                    st.success("Memory settings saved successfully!")
                except Exception as e:
                    st.error(f"Error saving memory settings: {str(e)}")

