import streamlit as st
from typing import Optional, Any, Dict, List
import os
from src.config import Config, MemoryConfig, KnowledgeConfig
from src.knowledge import KnowledgeBase

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
        StreamlitUI.setup_knowledge_config_ui()

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

    @staticmethod
    def setup_knowledge_config_ui():
        """Setup UI for knowledge base configuration"""
        with st.sidebar:
            st.markdown("### Knowledge Base")
            # Initialize knowledge base
            if 'knowledge_base' not in st.session_state:
                st.session_state.knowledge_base = KnowledgeBase()
            
            # Get knowledge sources
            sources = st.session_state.knowledge_base.get_sources()
            
            # Create tabs for different functions
            add_tab, view_tab = st.tabs(["Add Sources", "View Sources"])
            
            with add_tab:
                # Section for adding new knowledge sources
                st.markdown("#### Add Knowledge Source")
                
                # Source type selection
                source_type = st.selectbox(
                    "Source Type",
                    options=["PDF", "CSV", "JSON", "Text", "URL"],
                    key="knowledge_source_type"
                )
                
                # Common fields
                name = st.text_input("Source Name", key="knowledge_source_name", 
                                    placeholder="Enter a name for this source")
                description = st.text_area("Description", key="knowledge_source_description", 
                                         placeholder="Enter a description (optional)")
                
                # File upload or URL input based on source type
                if source_type in ["PDF", "CSV", "JSON"]:
                    uploaded_file = st.file_uploader(f"Upload {source_type} File", 
                                                   type=[source_type.lower()],
                                                   key=f"knowledge_file_{source_type.lower()}")
                    if st.button("Add Source", key=f"add_{source_type.lower()}_btn"):
                        if not name or not uploaded_file:
                            st.error("Name and file are required.")
                        else:
                            try:
                                file_content = uploaded_file.getvalue()
                                success = False
                                
                                if source_type == "PDF":
                                    success = st.session_state.knowledge_base.add_pdf_source(
                                        name, file_content, description)
                                    print(f"PDF source added: {name}, {description}")
                                elif source_type == "CSV":
                                    success = st.session_state.knowledge_base.add_csv_source(
                                        name, file_content, description)
                                elif source_type == "JSON":
                                    success = st.session_state.knowledge_base.add_json_source(
                                        name, file_content, description)
                                
                                if success:
                                    st.success(f"{source_type} source added successfully!")
                                else:
                                    st.error(f"Failed to add {source_type} source.")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                elif source_type == "Text":
                    text_content = st.text_area("Text Content", key="knowledge_text_content",
                                              placeholder="Enter text content")
                    if st.button("Add Source", key="add_text_btn"):
                        if not name or not text_content:
                            st.error("Name and text content are required.")
                        else:
                            try:
                                success = st.session_state.knowledge_base.add_text_source(
                                    name, text_content, description)
                                if success:
                                    st.success("Text source added successfully!")
                                else:
                                    st.error("Failed to add text source.")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                elif source_type == "URL":
                    url = st.text_input("URL", key="knowledge_url",
                                      placeholder="Enter URL (e.g., https://example.com/doc)")
                    if st.button("Add Source", key="add_url_btn"):
                        if not name or not url:
                            st.error("Name and URL are required.")
                        else:
                            try:
                                success = st.session_state.knowledge_base.add_url_source(
                                    name, url, description)
                                if success:
                                    st.success("URL source added successfully!")
                                else:
                                    st.error("Failed to add URL source.")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            
            with view_tab:
                # Section for viewing and managing existing sources
                st.markdown("#### Knowledge Sources")
                
                if not sources:
                    st.info("No knowledge sources added yet.")
                else:
                    for i, source in enumerate(sources):
                        # Instead of using nested expanders, use a container with styling
                        source_container = st.container(border=True)
                        with source_container:
                            st.markdown(f"**{source.name}** ({source.source_type.upper()})")
                            st.write(f"**Description:** {source.description or 'N/A'}")
                            
                            if source.source_type == "url":
                                st.write(f"**URL:** {source.source_url}")
                            else:
                                st.write(f"**File:** {os.path.basename(source.content_path)}")
                            
                            col1, col2 = st.columns(2)
                            # Enable/disable toggle - Add a unique key with index
                            with col1:
                                enabled = st.toggle("Enabled", value=source.enabled, 
                                                  key=f"toggle_{source.name}_{i}")
                                if enabled != source.enabled:
                                    st.session_state.knowledge_base.toggle_source(source.name, enabled)
                            
                            # Remove button - Add a unique key with index
                            with col2:
                                if st.button("Remove", key=f"remove_{source.name}_{i}"):
                                    if st.session_state.knowledge_base.remove_source(source.name):
                                        st.success(f"Removed {source.name}")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to remove {source.name}")
                
                # Refresh button
                if st.button("Refresh Sources"):
                    st.rerun()
            
            # Embedding model configuration for knowledge base
            st.markdown("#### Embedding Model Settings")
            
            # Get the current embedding model
            current_embedding_model = MemoryConfig.embedding_model
            embedding_options = ["sentence-transformers", "openai", "huggingface", "google"]
            try:
                embedding_index = embedding_options.index(current_embedding_model)
            except ValueError:
                embedding_index = 0
            
            embedding_model = st.selectbox(
                "Embedding Model",
                options=embedding_options,
                index=embedding_index
            )
            
            # Show Google-specific settings if Google is selected
            if embedding_model == "google":
                google_api_key = st.text_input(
                    "Google API Key",
                    value=KnowledgeConfig.google_api_key,
                    type="password"
                )
                google_model = st.text_input(
                    "Google Embedding Model",
                    value=KnowledgeConfig.google_embedding_model
                )
            
            if st.button("Save Knowledge Settings"):
                try:
                    # Update configuration
                    if "env_vars" not in st.session_state:
                        st.session_state.env_vars = {}
                    
                    # Update embedding model
                    st.session_state.env_vars["EMBEDDING_MODEL"] = embedding_model
                    
                    # Update Google-specific settings if selected
                    if embedding_model == "google":
                        st.session_state.env_vars["GOOGLE_API_KEY"] = google_api_key
                        st.session_state.env_vars["GOOGLE_EMBEDDING_MODEL"] = google_model
                    
                    st.success("Knowledge settings saved successfully!")
                except Exception as e:
                    st.error(f"Error saving knowledge settings: {str(e)}")

