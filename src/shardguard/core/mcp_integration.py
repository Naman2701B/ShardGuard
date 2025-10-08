"""MCP client integration for ShardGuard using the official Python SDK."""

import logging
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with MCP servers."""

    def __init__(self):
        """Initialize the MCP client."""
        import os

        # Get the absolute path to the servers directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        servers_dir = os.path.join(os.path.dirname(current_dir), "mcp_servers")

        self.server_configs = {
            "file-operations": {
                "command": sys.executable,
                "args": [os.path.join(servers_dir, "file_server.py")],
                "description": "File operations with security controls",
            },
            "email-operations": {
                "command": sys.executable,
                "args": [os.path.join(servers_dir, "email_server.py")],
                "description": "Email operations with privacy controls",
            },
            "database-operations": {
                "command": sys.executable,
                "args": [os.path.join(servers_dir, "database_server.py")],
                "description": "Database operations with security controls",
            },
            "web-operations": {
                "command": sys.executable,
                "args": [os.path.join(servers_dir, "web_server.py")],
                "description": "Web operations with security controls",
            },
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
            if hasattr(e, "exceptions"):
                logger.debug("  Sub-exceptions: %d", len(e.exceptions))
                for i, sub_e in enumerate(e.exceptions):
                    logger.debug("    %d: %s: %s", i, type(sub_e).__name__, sub_e)
            return None

    async def list_tools(self, server_name: str | None = None) -> dict[str, list[Any]]:
        """List available tools from one or all servers."""
        tools_by_server = {}

        servers_to_check = (
            [server_name] if server_name else list(self.server_configs.keys())
        )

        for server in servers_to_check:

            async def get_tools(session):
                tools_response = await session.list_tools()
                return tools_response.tools

            tools = await self._execute_with_server(server, get_tools)
            tools_by_server[server] = tools or []

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
                config = self.server_configs.get(server_name, {})
                server_desc = config.get("description", "MCP Server")
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

        return description

    def get_available_servers(self) -> dict[str, str]:
        """Get list of available servers and their descriptions."""
        return {
            name: config["description"] for name, config in self.server_configs.items()
        }
