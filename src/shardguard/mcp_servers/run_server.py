import asyncio
import argparse
import yaml

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Load tool configuration from YAML
def load_tools_from_yaml(server_name: str, path: str = None) -> list[Tool]:
    import os
    if path is None:
        # Automatically go to root folder where servers.yaml is expected
        current_file = os.path.abspath(__file__)
        root_dir = os.path.dirname(current_file)  # Go up to shardguard root
        path = os.path.join(root_dir, "servers.yaml")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    tools_data = config.get(server_name, [])
    tools = []
    for tool_data in tools_data:
        tools.append(
            Tool(
                name=tool_data["name"],
                description=tool_data["description"],
                inputSchema=tool_data["inputSchema"],
            )
        )
    return tools

# Main unified server function
def create_server(server_name: str):
    server = Server(server_name)
    tools = load_tools_from_yaml(server_name)

    # List tools from YAML
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tools

    # Generic handler with PoC response
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        args_repr = "\n".join(
            f"  {key}: {str(val)[:100]}{'...' if len(str(val)) > 100 else ''}"
            for key, val in arguments.items()
        )
        response = (
            f"[{server_name.upper()} PoC] Tool called: {name}\n"
            f"Arguments:\n{args_repr}"
        )
        return [TextContent(type="text", text=response)]

    return server

# Async main
async def main():
    parser = argparse.ArgumentParser(description="Start a unified MCP server.")
    parser.add_argument(
        "--server-name",
        type=str,
        required=True,
        help="Name of the server (e.g., blockchain-operations)",
    )
    args = parser.parse_args()

    server = create_server(args.server_name)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
