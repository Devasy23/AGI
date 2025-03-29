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


class OpenAIEmbedding(EmbeddingModel):
    """Embedding model using OpenAI's API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        try:
            from openai import OpenAI
            self.api_key = api_key or MemoryConfig.openai_api_key
            if not self.api_key:
                raise ValueError("OpenAI API key is required")
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Could not import openai. Please install it with `pip install openai`."
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using OpenAI API"""
        results = []
        # Process in batches to avoid token limits
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                input=batch_texts,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [e.embedding for e in response.data]
            results.extend(batch_embeddings)
        return results
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query string using OpenAI API"""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding


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
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "textembedding-gecko"):
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
        elif embedding_type == "openai":
            return OpenAIEmbedding(api_key=MemoryConfig.openai_api_key)
        elif embedding_type == "huggingface":
            return HuggingFaceEmbedding(api_key=MemoryConfig.hf_api_key)
        elif embedding_type == "google":
            return GoogleEmbedding(
                api_key=KnowledgeConfig.google_api_key, 
                model_name=KnowledgeConfig.google_embedding_model
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {embedding_type}")