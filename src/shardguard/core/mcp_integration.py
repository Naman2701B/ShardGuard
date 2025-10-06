"""MCP client integration for ShardGuard using the official Python SDK."""

import logging
import sys
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .yaml_mcp_server_factory import get_factory

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with MCP servers."""

    def __init__(self):
        # Get the YAML factory instance
        self.factory = get_factory()
        
        # Build server configs from YAML factory
        self.server_configs = {}
        available_servers = self.factory.get_available_servers()
        
        for server_name, description in available_servers.items():
            self.server_configs[server_name] = {
                "command": sys.executable,
                "args": [os.path.join(os.path.dirname(os.path.abspath(__file__)), "yaml_mcp_server_factory.py"), server_name],
                "description": description,
            }

    async def _execute_with_server(self, server_name: str, operation):
        """Execute an operation with a server connection."""
        if server_name not in self.server_configs:
            return None

        try:
            config = self.server_configs[server_name]
            server_params = StdioServerParameters(
                command=config["command"], args=config["args"]
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await operation(session)
        except Exception as e:
            logger.debug(
                "Error connecting to %s: %s: %s", server_name, type(e).__name__, e
            )
            if hasattr(e, "__cause__") and e.__cause__:
                logger.debug(
                    "  Caused by: %s: %s", type(e.__cause__).__name__, e.__cause__
                )
            exceptions = getattr(e, "exceptions", None)
            if exceptions:
                logger.debug("  Sub-exceptions: %d", len(exceptions))
                for i, sub_e in enumerate(exceptions):
                    logger.debug("    %d: %s: %s", i, type(sub_e).__name__, sub_e)
            return None

    async def list_tools(self, server_name: str | None = None) -> dict[str, list[Any]]:
        # Use the YAML factory to get tools directly
        if server_name:
            # Get tools for specific server
            tools_by_server = {}
            if server_name in self.server_configs:
                tools = self.factory.list_all_tools().get(server_name, [])
                tools_by_server[server_name] = tools
        else:
            # Get tools for all servers
            tools_by_server = self.factory.list_all_tools()
        
        return tools_by_server

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> str | None:
        """Call a tool on a specific server."""

        async def call_tool_op(session):
            result = await session.call_tool(tool_name, arguments)

            # Extract text content from the response
            if result.content:
                return "\n".join(
                    item.text for item in result.content if hasattr(item, "text")
                )
            return "Tool executed successfully (no content returned)"

        return await self._execute_with_server(server_name, call_tool_op)

    async def get_tools_description(self) -> str:
        """Get a formatted description of all available tools."""
        tools_by_server = await self.list_tools()

        if not any(tools_by_server.values()):
            return "No MCP tools available."

        description = "Available MCP Tools:\n\n"

        for server_name, tools in tools_by_server.items():
            if tools:
                server_desc = self.factory.get_server_config(server_name).get("description", "MCP Server")
                description += f"Server: {server_name} - {server_desc}\n"

                for tool in tools:
                    description += f"  • {tool.name}: {tool.description}\n"

                    # Add input schema details
                    if hasattr(tool, "inputSchema") and tool.inputSchema:
                        schema = tool.inputSchema
                        if isinstance(schema, dict) and "properties" in schema:
                            required = schema.get("required", [])
                            for prop_name, prop_info in schema["properties"].items():
                                req_marker = (
                                    " (required)" if prop_name in required else ""
                                )
                                prop_desc = prop_info.get(
                                    "description", "No description"
                                )
                                description += (
                                    f"    - {prop_name}: {prop_desc}{req_marker}\n"
                                )

                description += "\n"

        description += "When suggesting tools for tasks, include the tool names in your sub-task 'suggested_tools' field."
        return description

    def get_available_servers(self) -> dict[str, str]:
        return self.factory.get_available_servers() 
