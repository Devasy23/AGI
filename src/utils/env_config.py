from typing import Dict, Any
import os
from dotenv import load_dotenv
import streamlit as st

class EnvConfig:
    @staticmethod
    def load_env() -> Dict[str, Any]:
        """Load and validate environment variables"""
        load_dotenv()
        
        # Define required variables and their defaults
        env_vars = {
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "ollama"),
            "LLM_MODEL": os.getenv("LLM_MODEL", "gemma3:4b"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "")
        }
        
        # Validate provider-specific requirements
        if env_vars["LLM_PROVIDER"] == "groq" and not env_vars["GROQ_API_KEY"]:
            st.error("Groq API key is required when using Groq provider")
            raise ValueError("Missing GROQ_API_KEY")
            
        if env_vars["LLM_PROVIDER"] == "gemini" and not env_vars["GEMINI_API_KEY"]:
            st.error("Gemini API key is required when using Gemini provider")
            raise ValueError("Missing GEMINI_API_KEY")
        
        return env_vars
    
    @staticmethod
    def get_env(key: str, default: Any = None) -> Any:
        """Get a specific environment variable with optional default"""
        if key not in st.session_state.get("env_vars", {}):
            st.session_state.env_vars = EnvConfig.load_env()
        return st.session_state.env_vars.get(key, default)
    
    @staticmethod
    def setup_env_ui():
        """Create UI elements for environment configuration"""
        with st.sidebar:
            st.markdown("### Environment Configuration")
            
            # LLM Provider Selection
            provider = st.selectbox(
                "LLM Provider",
                options=["ollama", "groq", "gemini"],
                index=["ollama", "groq", "gemini"].index(EnvConfig.get_env("LLM_PROVIDER", "ollama"))
            )
            
            model = st.text_input("Enter Model Name", value=EnvConfig.get_env("LLM_MODEL", "gemma3:4b"))
            # Provider-specific settings
            if provider in ["groq", "gemini"]:
                api_key = st.text_input("API Key", value=EnvConfig.get_env("GROQ_API_KEY" if provider == "groq" else "GEMINI_API_KEY", ""), type
                ="password")

            
            # Save button
            if st.button("Save Configuration"):
                st.write(f"Debug out: {provider}, {model}, {api_key}")
                new_env = {
                    "LLM_PROVIDER": provider,
                    "LLM_MODEL": model if provider == "ollama" else EnvConfig.get_env("LLM_MODEL"),
                    "GROQ_API_KEY": api_key if provider == "groq" else EnvConfig.get_env("GROQ_API_KEY"),
                    "GEMINI_API_KEY": api_key if provider == "gemini" else EnvConfig.get_env("GEMINI_API_KEY")
                }
                st.session_state.env_vars = new_env
                st.write(".env file updated successfully")
                # Save to .env file
                with open(".env", "w") as f:
                    for key, value in new_env.items():
                        f.write(f"{key}={value}\n")
                st.success("Configuration saved!")