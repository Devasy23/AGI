import requests
from typing import Dict, Any, Optional
from src.config.mcp_config import MCPConfig

class MCPClient:
    """Client for interacting with Model Context Protocol server"""
    
    def __init__(self):
        self.config = MCPConfig.get_config()
        self.base_url = self.config["server_url"]
        self.headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
    
    def fetch_context(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch context from MCP server
        :param query: Query parameters for context fetch
        :return: Context data from server
        """
        response = requests.get(
            f"{self.base_url}/context",
            params=query,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def update_context(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update context on MCP server
        :param context_data: New context data to be stored
        :return: Response from server
        """
        response = requests.post(
            f"{self.base_url}/context",
            json=context_data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()