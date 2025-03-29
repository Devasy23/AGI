"""
Knowledge base implementation for CrewAI agents.
Supports various file types: PDF, CSV, JSON, raw text, and online docs.
"""
import os
import json
import tempfile
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import streamlit as st
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource

from src.config.knowledge_config import KnowledgeConfig
from .embeddings import EmbeddingFactory


class KnowledgeSource:
    """Represents a knowledge source with metadata"""
    
    def __init__(
        self, 
        source_type: str,
        name: str,
        content_path: str,
        description: str = "",
        source_url: str = "",
        enabled: bool = True
    ):
        """Initialize a knowledge source"""
        self.source_type = source_type  # pdf, csv, json, text, url
        self.name = name
        self.content_path = content_path
        self.description = description
        self.source_url = source_url
        self.enabled = enabled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "source_type": self.source_type,
            "name": self.name,
            "content_path": self.content_path,
            "description": self.description,
            "source_url": self.source_url,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeSource':
        """Create instance from dictionary"""
        return cls(
            source_type=data.get("source_type", ""),
            name=data.get("name", ""),
            content_path=data.get("content_path", ""),
            description=data.get("description", ""),
            source_url=data.get("source_url", ""),
            enabled=data.get("enabled", True)
        )


class KnowledgeBase:
    """
    Knowledge base for CrewAI agents.
    Manages sources and integration with CrewAI knowledge system.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the knowledge base
        
        Args:
            chunk_size: Size of chunks for processing large documents
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.knowledge_dir = KnowledgeConfig.knowledge_dir
        self.collection_name = KnowledgeConfig.collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sources: List[KnowledgeSource] = []
        self._load_sources()
    
    def _load_sources(self) -> None:
        """Load sources from index file"""
        index_path = os.path.join(self.knowledge_dir, "index.json")
        try:
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    sources_data = json.load(f)
                
                # Create sources with normalized paths
                self.sources = []
                for data in sources_data:
                    # Make sure paths are absolute and don't have duplicated prefixes
                    if data.get("source_type") != "url" and not os.path.isabs(data.get("content_path", "")):
                        # Convert to absolute path if it's a relative path
                        data["content_path"] = os.path.abspath(os.path.join(self.knowledge_dir, os.path.basename(data["content_path"])))
                    
                    self.sources.append(KnowledgeSource.from_dict(data))
        except Exception as e:
            st.error(f"Error loading knowledge sources: {str(e)}")
            self.sources = []
    
    def _save_sources(self) -> None:
        """Save sources to index file"""
        index_path = os.path.join(self.knowledge_dir, "index.json")
        try:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            with open(index_path, "w") as f:
                sources_data = [source.to_dict() for source in self.sources]
                json.dump(sources_data, f, indent=2)
        except Exception as e:
            st.error(f"Error saving knowledge sources: {str(e)}")
    
    def add_pdf_source(self, name: str, file_content: bytes, description: str = "") -> bool:
        """
        Add a PDF file as a knowledge source
        Args:
            name: Name of the knowledge source
            file_content: Content of the PDF file
            description: Optional description
        Returns:
            bool: Success status
        """
        try:
            # Ensure name has .pdf extension
            if not name.lower().endswith('.pdf'):
                name = f"{name}.pdf"
            
            # Create knowledge directory if it doesn't exist
            os.makedirs(self.knowledge_dir, exist_ok=True)
            
            # Use os.path.join for proper path construction
            file_path = os.path.join(self.knowledge_dir, name)
            
            # Write the PDF file
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Create source with relative path for storage
            source = KnowledgeSource(
                source_type="pdf",
                name=name,
                content_path=os.path.basename(file_path),  # Store only filename, not full path
                description=description
            )
            self.sources.append(source)
            self._save_sources()
            return True
        except Exception as e:
            st.error(f"Error adding PDF source: {str(e)}")
            return False
    
    def add_csv_source(self, name: str, file_content: bytes, description: str = "") -> bool:
        """
        Add a CSV file as a knowledge source
        Args:
            name: Name of the knowledge source
            file_content: Content of the CSV file
            description: Optional description
        Returns:
            bool: Success status
        """
        try:
            # Ensure name has .csv extension
            if not name.lower().endswith('.csv'):
                name = f"{name}.csv"
            
            # Create knowledge directory if it doesn't exist
            os.makedirs(self.knowledge_dir, exist_ok=True)
            
            # Use os.path.join for proper path construction
            file_path = os.path.join(self.knowledge_dir, name)
            
            # Write the CSV file
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Create source with relative path for storage
            source = KnowledgeSource(
                source_type="csv",
                name=name,
                content_path=os.path.basename(file_path),
                description=description
            )
            self.sources.append(source)
            self._save_sources()
            return True
        except Exception as e:
            st.error(f"Error adding CSV source: {str(e)}")
            return False
    
    def add_json_source(self, name: str, file_content: bytes, description: str = "") -> bool:
        """
        Add a JSON file as a knowledge source
        Args:
            name: Name of the knowledge source
            file_content: Content of the JSON file
            description: Optional description
        Returns:
            bool: Success status
        """
        try:
            # Ensure name has .json extension
            if not name.lower().endswith('.json'):
                name = f"{name}.json"
            
            # Create knowledge directory if it doesn't exist
            os.makedirs(self.knowledge_dir, exist_ok=True)
            
            # Use os.path.join for proper path construction
            file_path = os.path.join(self.knowledge_dir, name)
            
            # Write the JSON file
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Create source with relative path for storage
            source = KnowledgeSource(
                source_type="json",
                name=name,
                content_path=os.path.basename(file_path),
                description=description
            )
            self.sources.append(source)
            self._save_sources()
            return True
        except Exception as e:
            st.error(f"Error adding JSON source: {str(e)}")
            return False
    
    def add_text_source(self, name: str, content: str, description: str = "") -> bool:
        """
        Add raw text as a knowledge source
        Args:
            name: Name of the knowledge source
            content: Text content
            description: Optional description
        Returns:
            bool: Success status
        """
        try:
            # Use absolute path to ensure proper file location
            file_path = os.path.abspath(os.path.join(self.knowledge_dir, f"{name}.txt"))
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            source = KnowledgeSource(
                source_type="text",
                name=name,
                content_path=file_path,
                description=description
            )
            self.sources.append(source)
            self._save_sources()
            return True
        except Exception as e:
            st.error(f"Error adding text source: {str(e)}")
            return False
    
    def add_url_source(self, name: str, url: str, description: str = "") -> bool:
        """
        Add an online document URL as a knowledge source
        Args:
            name: Name of the knowledge source
            url: URL of the online document
            description: Optional description
        Returns:
            bool: Success status
        """
        try:
            source = KnowledgeSource(
                source_type="url",
                name=name,
                content_path=url,  # Store URL in content_path
                description=description,
                source_url=url
            )
            self.sources.append(source)
            self._save_sources()
            return True
        except Exception as e:
            st.error(f"Error adding URL source: {str(e)}")
            return False
    
    def remove_source(self, name: str) -> bool:
        """
        Remove a knowledge source by name
        Args:
            name: Name of the knowledge source to remove
        Returns:
            bool: Success status
        """
        try:
            for i, source in enumerate(self.sources):
                if source.name == name:
                    # Remove the file if it's stored locally
                    if source.source_type != "url" and os.path.exists(source.content_path):
                        try:
                            os.remove(source.content_path)
                        except:
                            pass
                    
                    # Remove from sources list
                    self.sources.pop(i)
                    self._save_sources()
                    return True
            return False
        except Exception as e:
            st.error(f"Error removing knowledge source: {str(e)}")
            return False
    
    def get_sources(self) -> List[KnowledgeSource]:
        """Get all knowledge sources"""
        return self.sources
    
    def get_enabled_sources(self) -> List[KnowledgeSource]:
        """Get only enabled knowledge sources"""
        return [source for source in self.sources if source.enabled]
    
    def toggle_source(self, name: str, enabled: bool) -> bool:
        """
        Toggle a knowledge source's enabled status
        Args:
            name: Name of the knowledge source
            enabled: New enabled status
        Returns:
            bool: Success status
        """
        try:
            for source in self.sources:
                if source.name == name:
                    source.enabled = enabled
                    self._save_sources()
                    return True
            return False
        except Exception as e:
            st.error(f"Error toggling knowledge source: {str(e)}")
            return False
    
    def get_crew_knowledge_sources(self) -> List[Any]:
        """
        Convert knowledge sources to CrewAI compatible knowledge sources
        Returns:
            List of CrewAI knowledge source objects
        """
        crew_sources = []
        
        for source in self.get_enabled_sources():
            try:
                # For debugging
                st.write(f"Processing source: {source.name} ({source.source_type})")
                
                # For PDF sources, just use the filename without path
                # (file should already be copied to working directory by CrewWorkflow._prepare_knowledge_files)
                if source.source_type == "pdf":
                    filename = os.path.basename(source.content_path)
                    st.write(f"Using PDF with filename only: {filename}")
                    
                    # Verify the file exists in current working directory
                    working_dir_path = os.path.join(os.getcwd(), filename)
                    if os.path.exists(working_dir_path):
                        crew_sources.append(PDFKnowledgeSource(file_paths=[filename]))
                        st.write(f"Successfully added PDF source: {source.name}")
                    else:
                        st.error(f"PDF file not found in working directory: {working_dir_path}")
                        
                # Handle other file types
                elif source.source_type in ["csv", "json", "text"]:
                    # For other file types, use the full path
                    if os.path.exists(source.content_path):
                        if source.source_type == "csv":
                            crew_sources.append(CSVKnowledgeSource(file_paths=[source.content_path]))
                        elif source.source_type == "json":
                            crew_sources.append(JSONKnowledgeSource(file_paths=[source.content_path]))
                        elif source.source_type == "text":
                            crew_sources.append(TextFileKnowledgeSource(file_paths=[source.content_path]))
                        st.write(f"Successfully added {source.source_type.upper()} source: {source.name}")
                    else:
                        st.error(f"File not found: {source.content_path}")
                
                # Handle URL sources
                elif source.source_type == "url":
                    crew_sources.append(CrewDoclingSource(file_paths=[source.content_path]))
                    st.write(f"Successfully added URL source: {source.name}")
                
            except Exception as e:
                st.error(f"Error loading knowledge source {source.name}: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        return crew_sources
    
    def create_crew_knowledge(self) -> Optional[Knowledge]:
        """
        Create a CrewAI Knowledge instance with all enabled sources
        Returns:
            Knowledge instance if sources exist, None otherwise
        """
        sources = self.get_crew_knowledge_sources()
        
        if not sources:
            return None
        
        return Knowledge(
            sources=sources,
            collection_name=self.collection_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def reset_memories(self) -> bool:
        """
        Reset the knowledge base memories.
        Useful when knowledge sources have been updated.
        
        Returns:
            bool: Success status
        """
        try:
            knowledge = self.create_crew_knowledge()
            if knowledge:
                knowledge.reset_memories()
            return True
        except Exception as e:
            st.error(f"Error resetting memories: {str(e)}")
            return False