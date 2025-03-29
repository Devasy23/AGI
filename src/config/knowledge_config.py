from typing import Literal, Optional, List
import os
from dotenv import load_dotenv
import streamlit as st
from .memory_config import MemoryConfig

# Load environment variables
load_dotenv()

class KnowledgeConfig:
    @staticmethod
    def clean_env_value(value: str) -> str:
        """Clean environment variable value by removing comments and whitespace"""
        if value:
            return value.split('#')[0].strip()
        return value or ""
    
    # Knowledge base settings
    knowledge_dir: str = clean_env_value(os.getenv("KNOWLEDGE_DIR", "./knowledge"))
    collection_name: str = clean_env_value(os.getenv("KNOWLEDGE_COLLECTION", "knowledge"))
    
    # Default settings for Google's embedding model
    google_embedding_model: str = clean_env_value(os.getenv("GOOGLE_EMBEDDING_MODEL", "textembedding-gecko"))
    google_api_key: str = clean_env_value(os.getenv("GOOGLE_API_KEY", ""))
    
    @classmethod
    def get_embedding_api_key(cls):
        """Get the appropriate API key based on the selected embedding model"""
        embedding_model = MemoryConfig.embedding_model
        
        if embedding_model == "openai":
            return MemoryConfig.openai_api_key
        elif embedding_model == "huggingface":
            return MemoryConfig.hf_api_key
        elif embedding_model == "google":
            if 'env_vars' in st.session_state and 'GOOGLE_API_KEY' in st.session_state.env_vars:
                return st.session_state.env_vars['GOOGLE_API_KEY']
            return cls.google_api_key
        else:
            return None
    
    @classmethod
    def get_config(cls):
        """Get all knowledge configuration settings"""
        return {
            "knowledge_dir": cls.knowledge_dir,
            "collection_name": cls.collection_name,
            "google_embedding_model": cls.google_embedding_model,
            "google_api_key": cls.google_api_key
        }
    
    @classmethod
    def validate_config(cls):
        """Validate the knowledge configuration"""
        config = cls.get_config()
        
        # Create knowledge directory if it doesn't exist
        os.makedirs(config["knowledge_dir"], exist_ok=True)
        
        # Validate Google API key if using Google embeddings
        if MemoryConfig.embedding_model == "google" and not cls.get_embedding_api_key():
            raise ValueError("Google API key is required when using Google embeddings")