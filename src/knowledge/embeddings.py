"""
Module for managing embeddings in the knowledge base system.
"""
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import os
from src.config.memory_config import MemoryConfig
from src.config.knowledge_config import KnowledgeConfig

class EmbeddingModel(ABC):
    """Base abstract class for embedding models"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convert list of texts into embeddings"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Convert a query text into an embedding"""
        pass


class SentenceTransformersEmbedding(EmbeddingModel):
    """Embedding model using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a specific model name"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "Could not import sentence_transformers. "
                "Please install it with `pip install sentence-transformers`."
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string"""
        return self.model.encode(text, convert_to_numpy=True).tolist()


class HuggingFaceEmbedding(EmbeddingModel):
    """Embedding model using HuggingFace's API"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with API key and model name"""
        try:
            import requests
            self.requests = requests
            self.api_key = api_key or MemoryConfig.hf_api_key
            if not self.api_key:
                raise ValueError("HuggingFace API key is required")
            self.model_name = model_name
            self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
            self.headers = {"Authorization": f"Bearer {self.api_key}"}
        except ImportError:
            raise ImportError(
                "Could not import requests. Please install it with `pip install requests`."
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using HuggingFace API"""
        results = []
        # Process one at a time to avoid API limits
        for text in texts:
            response = self.requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": text}
            )
            response.raise_for_status()
            results.append(response.json())
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string using HuggingFace API"""
        response = self.requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text}
        )
        response.raise_for_status()
        return response.json()


class GoogleEmbedding(EmbeddingModel):
    """Embedding model using Google's Embedding API"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/text-embedding-004"):
        """Initialize with API key and model name"""
        try:
            import google.generativeai as genai
            self.api_key = api_key or KnowledgeConfig.google_api_key
            if not self.api_key:
                raise ValueError("Google API key is required")
            self.model_name = model_name
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "Could not import google-generativeai. "
                "Please install it with `pip install google-generativeai`."
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Google's API"""
        results = []
        for text in texts:
            embedding = self.genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            if hasattr(embedding, 'embedding'):
                results.append(embedding.embedding)
            else:
                results.append(embedding['embedding'])
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string using Google's API"""
        embedding = self.genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        if hasattr(embedding, 'embedding'):
            return embedding.embedding
        return embedding['embedding']


class EmbeddingFactory:
    """Factory for creating embedding model instances"""
    
    @staticmethod
    def get_embedding_model() -> EmbeddingModel:
        """Create and return an embedding model based on configuration"""
        embedding_type = MemoryConfig.embedding_model
        
        if embedding_type == "sentence-transformers":
            return SentenceTransformersEmbedding(model_name=MemoryConfig.embedding_model_name)
        elif embedding_type == "huggingface":
            return HuggingFaceEmbedding(api_key=MemoryConfig.hf_api_key)
        elif embedding_type == "google":
            return GoogleEmbedding(
                api_key=KnowledgeConfig.google_api_key, 
                model_name=KnowledgeConfig.google_embedding_model
            )
        else:
            # Default to sentence-transformers if type not recognized
            print(f"Embedding type '{embedding_type}' not recognized, using sentence-transformers instead")
            return SentenceTransformersEmbedding(model_name="all-MiniLM-L6-v2")