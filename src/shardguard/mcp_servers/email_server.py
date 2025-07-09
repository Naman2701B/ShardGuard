"""Email operations MCP server for ShardGuard."""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Create the MCP server instance
server = Server("email-operations")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available email operation tools."""
    return [
        Tool(
            name="send_email",
            description="Send an email (PoC: prints action)",
            inputSchema={
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Email recipient address",
                    },
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body": {"type": "string", "description": "Email body content"},
                },
                "required": ["recipient", "subject", "body"],
            },
        ),
        Tool(
            name="list_emails",
            description="List recent emails (PoC: prints action)",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of emails to list",
                        "default": 10,
                    }
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "send_email":
        recipient = arguments["recipient"]
        subject = arguments["subject"]
        body = arguments["body"]
        result = "[EMAIL PoC] Would send email:\n"
        result += f"  To: {recipient}\n"
        result += f"  Subject: {subject}\n"
        result += f"  Body: {body[:100]}{'...' if len(body) > 100 else ''}"
        return [TextContent(type="text", text=result)]

    elif name == "list_emails":
        limit = arguments.get("limit", 10)
        result = f"[EMAIL PoC] Would list {limit} recent emails"
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
