"""Web operations MCP server for ShardGuard."""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Create the MCP server instance
server = Server("web-operations")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available web operation tools."""
    return [
        Tool(
            name="http_request",
            description="Make an HTTP request (PoC: prints action)",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to make request to"},
                    "method": {
                        "type": "string",
                        "description": "HTTP method",
                        "default": "GET",
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers as key-value pairs",
                        "default": {},
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body content",
                        "default": "",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="scrape_website",
            description="Scrape content from a website (PoC: prints action)",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for specific content",
                        "default": "",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="download_file",
            description="Download a file from the web (PoC: prints action)",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of file to download"},
                    "filename": {
                        "type": "string",
                        "description": "Local filename to save as",
                    },
                },
                "required": ["url", "filename"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "http_request":
        url = arguments["url"]
        method = arguments.get("method", "GET")
        headers = arguments.get("headers", {})
        body = arguments.get("body", "")

        result = "[WEB PoC] Would make HTTP request:\n"
        result += f"  URL: {url}\n"
        result += f"  Method: {method}\n"
        if headers:
            result += f"  Headers: {len(headers)} headers\n"
        if body:
            result += f"  Body: {body[:50]}{'...' if len(body) > 50 else ''}"

        return [TextContent(type="text", text=result)]

    elif name == "scrape_website":
        url = arguments["url"]
        selector = arguments.get("selector", "")

        result = "[WEB PoC] Would scrape website:\n"
        result += f"  URL: {url}\n"
        if selector:
            result += f"  Selector: {selector}"

        return [TextContent(type="text", text=result)]

    elif name == "download_file":
        url = arguments["url"]
        filename = arguments["filename"]

        result = "[WEB PoC] Would download file:\n"
        result += f"  URL: {url}\n"
        result += f"  Save as: {filename}"

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
