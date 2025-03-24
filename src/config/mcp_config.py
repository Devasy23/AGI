from typing import Literal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MCPConfig:
    """Configuration for Model Context Protocol"""
    
    # MCP Server configuration
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    mcp_api_key: str = os.getenv("MCP_API_KEY", "")
    
    @classmethod
    def get_config(cls):
        """Get MCP configuration"""
        return {
            "server_url": cls.mcp_server_url,
            "api_key": cls.mcp_api_key
        }
    
    @classmethod
    def validate_config(cls):
        """Validate the MCP configuration"""
        config = cls.get_config()
        
        if not config["server_url"]:
            raise ValueError("MCP server URL is required")
            
        if not config["api_key"]:
            raise ValueError("MCP API key is required")