"""
Dynamic MCP Server Factory

This module creates MCP server instances dynamically from YAML configurations,
eliminating the need for separate Python files for each server.
"""

import asyncio
import sys
import os
from typing import Any, Dict, List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from .parse_fake_mcp_server_yaml import parse_fake_MCP_YAML_directory


class YAMLMCPServerFactory:
    """Factory for creating MCP servers from YAML configurations."""
    
    def __init__(self, yaml_directory: str = None):
        """
        Initialize the factory with a directory containing YAML server configurations.
        
        Args:
            yaml_directory: Path to directory containing YAML files. 
                          If None, defaults to 'mcp_servers' subdirectory.
        """
        if yaml_directory is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_directory = os.path.join(current_dir, "../mcp_servers")
        
        self.yaml_directory = yaml_directory
        self.server_configs = {}
        self._load_server_configs()
    
    def _load_server_configs(self):
        """Load all server configurations from YAML files."""
        # Added the following try block to loop through even the junk servers - which were being used for testing
        # This can later be removed if not needed OR we can even keep it to keep the logic intact so that we can create
        # more classified way of storing servers on the system. E.g: File Operation based servers in one folder, 
        # Nuke based in one and so on.
        try:
            merged: Dict[str, Dict[str, Any]] = {}
            if not os.path.isdir(self.yaml_directory):
                raise FileNotFoundError(f"YAML directory not found: {self.yaml_directory}")

            # Walk every subdir and call the existing parser on each directory.
            for root, dirs, files in os.walk(self.yaml_directory):
                if not any(fn.lower().endswith((".yml", ".yaml")) for fn in files):
                    continue

                try:
                    configs = parse_fake_MCP_YAML_directory(root)
                except Exception as suberr:
                    print(f"Warning: Skipping '{root}' due to parse error: {suberr}")
                    continue

                for cfg in configs:
                    name = cfg.get("server")
                    if not name:
                        continue
                    merged[name] = cfg

            self.server_configs = merged
            print(f"[MCP] Loaded {len(self.server_configs)} server config(s) from '{self.yaml_directory}' (recursive).")

        except Exception as e:
            print(f"Warning: Could not load YAML configurations: {e}")
            self.server_configs = {}
    
    def create_server_instance(self, server_name: str) -> Server:
        """
        Create an MCP server instance from YAML configuration.
        
        Args:
            server_name: Name of the server to create
            
        Returns:
            Configured MCP Server instance
            
        Raises:
            ValueError: If server_name is not found in configurations
        """
        if server_name not in self.server_configs:
            raise ValueError(f"Server '{server_name}' not found in YAML configurations")
        
        config = self.server_configs[server_name]
        
        # Create the server instance
        server = Server(server_name)
        
        # Convert YAML tools to MCP Tool objects
        tools = []
        for tool_config in config["tools"]:
            # Build input schema from properties
            properties = {}
            required = []
            
            for prop in tool_config["properties"]:
                prop_name = prop["name"]
                prop_type = prop["type"]
                prop_desc = prop["description"]
                
                # Convert YAML types to JSON schema types
                json_type = self._convert_type_to_json_schema(prop_type)
                
                properties[prop_name] = {
                    "type": json_type,
                    "description": prop_desc
                }
                
                # For now, make all properties required
                required.append(prop_name)
            
            input_schema = {
                "type": "object",
                "properties": properties,
                "required": required
            }
            
            tool = Tool(
                name=tool_config["name"],
                description=tool_config["description"],
                inputSchema=input_schema
            )
            tools.append(tool)
        
        # Register the list_tools handler
        @server.list_tools()
        async def list_tools() -> List[Tool]:
            return tools
        
        # Register the call_tool handler
        @server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            return await self._handle_tool_call(server_name, name, arguments, tools)
        
        return server
    
    def _convert_type_to_json_schema(self, yaml_type: str) -> str:
        """Convert YAML property types to JSON schema types."""
        type_mapping = {
            "string": "string",
            "number": "number", 
            "boolean": "boolean",
            "null": "null",
            "array": "array"
        }
        return type_mapping.get(yaml_type, "string")
    
    async def _handle_tool_call(self, server_name: str, tool_name: str, 
                               arguments: Dict[str, Any], tools: List[Tool]) -> List[TextContent]:
        """
        Handle tool calls for YAML-based servers.
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool
            tools: List of available tools for this server
            
        Returns:
            List of TextContent responses
        """
        # Find the tool configuration
        tool_config = None
        for tool in tools:
            if tool.name == tool_name:
                tool_config = tool
                break
        
        if not tool_config:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Generate a simulated response based on the tool and arguments
        response_text = self._generate_simulated_response(server_name, tool_name, arguments)
        
        return [TextContent(type="text", text=response_text)]
    
    def _generate_simulated_response(self, server_name: str, tool_name: str, 
                                   arguments: Dict[str, Any]) -> str:
        """
        Generate a simulated response for tool calls.
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Arguments passed to the tool
            
        Returns:
            Simulated response text
        """
        # Get server description for context
        server_desc = self.server_configs.get(server_name, {}).get("description", "MCP Server")
        
        # Format arguments for display
        args_str = ", ".join([f"{k}={v}" for k, v in arguments.items()])
        
        # Generate response based on server type
        if "nuke" in server_name.lower():
            return f"[NUKE PoC] {tool_name} executed:\n  Args: {args_str}\n  Server: {server_desc}\n  Simulation completed"
        elif "water" in server_name.lower():
            return f"[WATER PoC] {tool_name} executed:\n  Args: {args_str}\n  Server: {server_desc}\n  Simulation completed"
        elif "blockchain" in server_name.lower():
            return f"[BLOCKCHAIN PoC] {tool_name} executed:\n  Args: {args_str}\n  Server: {server_desc}\n  Simulation completed"
        elif "global" in server_name.lower():
            return f"[GLOBAL PoC] {tool_name} executed:\n  Args: {args_str}\n  Server: {server_desc}\n  Simulation completed"
        else:
            return f"[{server_name.upper()} PoC] {tool_name} executed:\n  Args: {args_str}\n  Server: {server_desc}\n  Simulation completed"
    
    def get_available_servers(self) -> Dict[str, str]:
        """Get dictionary of available server names and descriptions."""
        return {
            name: config["description"] 
            for name, config in self.server_configs.items()
        }
    
    def get_server_config(self, server_name: str) -> Dict[str, Any]:
        """Get the configuration for a specific server."""
        return self.server_configs.get(server_name, {})
    
    def list_all_tools(self) -> Dict[str, List[Tool]]:
        """Get all tools from all servers."""
        tools_by_server = {}
        for server_name in self.server_configs.keys():
            server = self.create_server_instance(server_name)
            # We need to get the tools from the server's list_tools handler
            # This is a bit tricky since we need to call the async handler
            # For now, we'll create the tools directly from the config
            config = self.server_configs[server_name]
            tools = []
            for tool_config in config["tools"]:
                properties = {}
                required = []
                
                for prop in tool_config["properties"]:
                    prop_name = prop["name"]
                    prop_type = prop["type"]
                    prop_desc = prop["description"]
                    
                    json_type = self._convert_type_to_json_schema(prop_type)
                    
                    properties[prop_name] = {
                        "type": json_type,
                        "description": prop_desc
                    }
                    required.append(prop_name)
                
                input_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
                
                tool = Tool(
                    name=tool_config["name"],
                    description=tool_config["description"],
                    inputSchema=input_schema
                )
                tools.append(tool)
            
            tools_by_server[server_name] = tools
        
        return tools_by_server


# Global factory instance
_factory = None

def get_factory() -> YAMLMCPServerFactory:
    """Get the global factory instance."""
    global _factory
    if _factory is None:
        _factory = YAMLMCPServerFactory()
    return _factory


async def run_server(server_name: str):
    """Run a specific server instance."""
    factory = get_factory()
    server = factory.create_server_instance(server_name)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    # Allow running a specific server from command line
    if len(sys.argv) > 1:
        server_name = sys.argv[1]
        asyncio.run(run_server(server_name))
    else:
        print("Usage: python yaml_mcp_server_factory.py <server_name>") 