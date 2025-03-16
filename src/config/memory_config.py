from typing import Literal
import os
from dotenv import load_dotenv

class MemoryConfig:
    vector_store: Literal["chroma", "qdrant", "faiss"] = os.getenv("VECTOR_STORE", "chroma")
    embedding_model: Literal["sentence-transformers", "openai", "huggingface"] = os.getenv("EMBEDDING_MODEL", "sentence-transformers")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # Vector store specific settings
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    
    # Embedding API keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    hf_api_key: str = os.getenv("HF_API_KEY", "")
    
    @classmethod
    def get_config(cls):
        return {
            "vector_store": cls.vector_store,
            "embedding_model": cls.embedding_model,
            "embedding_model_name": cls.embedding_model_name,
            "chroma_persist_dir": cls.chroma_persist_dir,
            "qdrant_url": cls.qdrant_url,
            "qdrant_api_key": cls.qdrant_api_key,
            "openai_api_key": cls.openai_api_key,
            "hf_api_key": cls.hf_api_key
        }
        
    @classmethod
    def validate_config(cls):
        """Validate the memory configuration"""
        config = cls.get_config()
        
        if config["vector_store"] == "qdrant" and not (config["qdrant_url"] and config["qdrant_api_key"]):
            raise ValueError("Qdrant URL and API key are required when using Qdrant vector store")
            
        if config["embedding_model"] == "openai" and not config["openai_api_key"]:
            raise ValueError("OpenAI API key is required when using OpenAI embeddings")
            
        if config["embedding_model"] == "huggingface" and not config["hf_api_key"]:
            raise ValueError("HuggingFace API key is required when using HuggingFace embeddings")