"""Database operations MCP server for ShardGuard."""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Create the MCP server instance
server = Server("database-operations")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available database operation tools."""
    return [
        Tool(
            name="query_database",
            description="Execute a database query (PoC: prints action)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {
                        "type": "string",
                        "description": "Database name",
                        "default": "default",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="backup_database",
            description="Create a database backup (PoC: prints action)",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name to backup",
                    },
                    "backup_name": {
                        "type": "string",
                        "description": "Name for the backup file",
                    },
                },
                "required": ["database", "backup_name"],
            },
        ),
        Tool(
            name="list_tables",
            description="List tables in a database (PoC: prints action)",
            inputSchema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Database name to list tables from",
                    }
                },
                "required": ["database"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "query_database":
        query = arguments["query"]
        database = arguments.get("database", "default")
        result = "[DATABASE PoC] Would execute query:\n"
        result += f"  Database: {database}\n"
        result += f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}"
        return [TextContent(type="text", text=result)]

    elif name == "backup_database":
        database = arguments["database"]
        backup_name = arguments["backup_name"]
        result = "[DATABASE PoC] Would backup database:\n"
        result += f"  Database: {database}\n"
        result += f"  Backup file: {backup_name}"
        return [TextContent(type="text", text=result)]

    elif name == "list_tables":
        database = arguments["database"]
        result = f"[DATABASE PoC] Would list tables in database: {database}"
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
