"""File operations MCP server for ShardGuard."""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Create the MCP server instance
server = Server("file-operations")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available file operation tools."""
    return [
        Tool(
            name="read_file",
            description="Read contents from a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write_file",
            description="Write content to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="list_directory",
            description="List contents of a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list",
                    }
                },
                "required": ["path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "read_file":
        path = arguments["path"]
        result = f"[FILE PoC] Would read file: {path}"
        return [TextContent(type="text", text=result)]

    elif name == "write_file":
        path = arguments["path"]
        content = arguments["content"]
        result = "[FILE PoC] Would write to file:\n"
        result += f"  Path: {path}\n"
        result += f"  Content: {content[:100]}{'...' if len(content) > 100 else ''}"
        return [TextContent(type="text", text=result)]

    elif name == "list_directory":
        path = arguments["path"]
        result = f"[FILE PoC] Would list directory: {path}"
        return [TextContent(type="text", text=result)]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
