from typing import Dict, Any
import os
from dotenv import load_dotenv
import streamlit as st

class EnvConfig:
    @staticmethod
    def load_env() -> Dict[str, Any]:
        """Load and validate environment variables"""
        load_dotenv()
        
        # Load from .env with defaults
        env_vars = {
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "ollama"),
            "LLM_MODEL": os.getenv("LLM_MODEL", "gemma3:4b"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "")
        }
        return env_vars

    @staticmethod
    def init_session_state():
        """Initialize session state with environment variables"""
        if not hasattr(st.session_state, 'env_vars'):
            st.session_state.env_vars = EnvConfig.load_env()
    
    @staticmethod
    def get_env(key: str, default: Any = None) -> Any:
        """Get a specific environment variable with optional default"""
        EnvConfig.init_session_state()
        return st.session_state.env_vars.get(key, default)
    
    @staticmethod
    def setup_env_ui():
        """Create UI elements for environment configuration"""
        EnvConfig.init_session_state()
        
        with st.sidebar:
            st.markdown("### Environment Configuration")
            
            current_provider = st.session_state.env_vars.get("LLM_PROVIDER", "ollama")
            current_model = st.session_state.env_vars.get("LLM_MODEL", "gemma3:4b")
            
            # LLM Provider Selection
            provider = st.selectbox(
                "LLM Provider",
                options=["ollama", "groq", "gemini"],
                index=["ollama", "groq", "gemini"].index(current_provider)
            )
            
            # Provider-specific settings
            if provider == "ollama":
                model = st.text_input("Ollama Model", value=current_model)
            elif provider == "groq":
                model = st.text_input("Groq Model", value=current_model)
                api_key = st.text_input("Groq API Key", 
                                      value=st.session_state.env_vars.get("GROQ_API_KEY", ""), 
                                      type="password")
            elif provider == "gemini":
                model = st.text_input("Gemini Model", value=current_model)
                api_key = st.text_input("Gemini API Key", 
                                      value=st.session_state.env_vars.get("GEMINI_API_KEY", ""), 
                                      type="password")
            
            if st.button("Save Configuration"):
                new_config = {
                    "LLM_PROVIDER": provider,
                    "LLM_MODEL": model,
                }
                
                if provider in ["groq", "gemini"]:
                    new_config[f"{provider.upper()}_API_KEY"] = api_key
                
                st.session_state.env_vars.update(new_config)
                st.success("Configuration updated!")
                # st.rerun()