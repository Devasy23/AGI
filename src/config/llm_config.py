from typing import Literal
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class LLMConfig:
    @classmethod
    def get_provider(cls):
        # First check session state, then environment variables, then default
        if 'env_vars' in st.session_state and 'LLM_PROVIDER' in st.session_state.env_vars:
            return st.session_state.env_vars['LLM_PROVIDER']
        return os.getenv("LLM_PROVIDER", "ollama")
    
    @classmethod
    def get_model_name(cls):
        if 'env_vars' in st.session_state and 'LLM_MODEL' in st.session_state.env_vars:
            return st.session_state.env_vars['LLM_MODEL']
        return os.getenv("LLM_MODEL", "gemma3:4b")
    
    @classmethod
    def get_groq_api_key(cls):
        if 'env_vars' in st.session_state and 'GROQ_API_KEY' in st.session_state.env_vars:
            return st.session_state.env_vars['GROQ_API_KEY']
        return os.getenv("GROQ_API_KEY", "")
    
    @classmethod
    def get_gemini_api_key(cls):
        if 'env_vars' in st.session_state and 'GEMINI_API_KEY' in st.session_state.env_vars:
            return st.session_state.env_vars['GEMINI_API_KEY']
        return os.getenv("GEMINI_API_KEY", "")
    
    @classmethod
    def get_config(cls):
        return {
            "provider": cls.get_provider(),
            "model_name": cls.get_model_name(),
            "groq_api_key": cls.get_groq_api_key(),
            "gemini_api_key": cls.get_gemini_api_key()
        }