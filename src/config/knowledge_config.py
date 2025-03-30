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
    
    # Chunking configuration
    chunk_size: int = int(clean_env_value(os.getenv("KNOWLEDGE_CHUNK_SIZE", "500")))
    chunk_overlap: int = int(clean_env_value(os.getenv("KNOWLEDGE_CHUNK_OVERLAP", "50")))
    
    # Default settings for Google's embedding model
    google_embedding_model: str = clean_env_value(os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004"))
    google_api_key: str = clean_env_value(os.getenv("GEMINI_API_KEY", ""))  # Use GEMINI_API_KEY by default

    @classmethod
    def get_embedding_api_key(cls):
        """Get the appropriate API key based on the selected embedding model"""
        embedding_model = MemoryConfig.embedding_model
        
        if embedding_model == "huggingface":
            return MemoryConfig.hf_api_key
        elif embedding_model == "google":
            # Prefer GEMINI_API_KEY from env vars if available
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                return gemini_key
            # Fall back to session state or config
            if 'env_vars' in st.session_state and 'GEMINI_API_KEY' in st.session_state.env_vars:
                return st.session_state.env_vars['GEMINI_API_KEY']
            return cls.google_api_key
        else:
            return None

    @classmethod
    def get_config(cls):
        """Get all knowledge configuration settings"""
        return {
            "knowledge_dir": cls.knowledge_dir,
            "collection_name": cls.collection_name,
            "chunk_size": cls.chunk_size,
            "chunk_overlap": cls.chunk_overlap,
            "google_embedding_model": cls.google_embedding_model,
            "google_api_key": cls.google_api_key
        }

    @classmethod
    def validate_config(cls):
        """Validate the knowledge configuration"""
        config = cls.get_config()
        
        # Create knowledge directory if it doesn't exist
        os.makedirs(config["knowledge_dir"], exist_ok=True)
        
        # Validate chunking parameters
        if config["chunk_size"] <= 0:
            raise ValueError("Chunk size must be positive")
        if config["chunk_overlap"] < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if config["chunk_overlap"] >= config["chunk_size"]:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        
        # Validate knowledge directory exists and is writeable
        if not os.path.exists(config["knowledge_dir"]):
            raise ValueError(f"Knowledge directory does not exist: {config['knowledge_dir']}")
        if not os.access(config["knowledge_dir"], os.W_OK):
            raise ValueError(f"Knowledge directory is not writeable: {config['knowledge_dir']}")
            
        # Validate Google API key if using Google embeddings
        if MemoryConfig.embedding_model == "google" and not cls.get_embedding_api_key():
            raise ValueError("Google API key is required when using Google embeddings")