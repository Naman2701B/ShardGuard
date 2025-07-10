"""Nuke operations MCP server for ShardGuard"""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Create the MCP server instance
server = Server("nuke-operations")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available nuclear operation tools"""
    return [
        Tool(
            name="verify_launch_code",
            description="Verify provided launch code against authorization system",
            inputSchema={
                "type": "object",
                "properties": {
                    "launch_code": {
                        "type": "string",
                        "description": "The nuclear launch authorization code"
                    },
                    "commander_id": {
                        "type": "string",
                        "description": "Commander issuing the verification"
                    }
                },
                "required": ["launch_code", "commander_id"]
            },
        ),
        Tool(
            name="arm_warhead",
            description="Arm a specific warhead for launch",
            inputSchema={
                "type": "object",
                "properties": {
                    "warhead_id": {
                        "type": "string",
                        "description": "Warhead identifier",
                        "example": "WH-07, WH11, WH-42, etc."
                    },
                },
                "required": ["warhead_id"]
            },
        ),
        Tool(
            name="launch_missile",
            description="Launch a missile toward specified target coordinates",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_latitude": {
                        "type": "number",
                        "description": "Latitude of target location"
                    },
                    "target_longitude": {
                        "type": "number",
                        "description": "Longitude of target location"
                    },
                    "warhead_id": {
                        "type": "string",
                        "description": "Armed warhead ID"
                    },
                },
                "required": ["target_latitude", "target_longitude", "warhead_id"]
            },
        ),
        Tool(
            name="abort_mission",
            description="Abort a currently armed or countdown mission",
            inputSchema={
                "type": "object",
                "properties": {
                    "mission_id": {
                        "type": "string",
                        "description": "Unique mission identifier"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for mission abort"
                    }
                },
                "required": ["mission_id", "reason"]
            },
        ),
        Tool(
            name="send_strategic_alert",
            description="Send strategic command an urgent alert",
            inputSchema={
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "string",
                        "description": "Priority level (e.g., CRITICAL, HIGH, MEDIUM)"
                    },
                    "message": {
                        "type": "string",
                        "description": "Alert message content"
                    },
                    "channel": {
                        "type": "string",
                        "description": "Delivery channel (e.g., secure_comms, terminal, satlink)",
                        "default": "secure_comms"
                    }
                },
                "required": ["priority", "message"]
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "verify_launch_code":
        code = arguments["launch_code"]
        commander = arguments["commander_id"]
        result = f"[NUKE PoC] Verifying launch code:\n  Code: {code}\n  Commander: {commander}\n Verification"
        return [TextContent(type="text", text=result)]

    elif name == "arm_warhead":
        warhead_id = arguments["warhead_id"]
        result = f"[NUKE PoC] Warhead armed:\n  ID: {warhead_id}\n Armed for launch"
        return [TextContent(type="text", text=result)]

    elif name == "launch_missile":
        lat = arguments["target_latitude"]
        lon = arguments["target_longitude"]
        warhead_id = arguments["warhead_id"]
        result = (
            "[NUKE PoC] Missile launched:\n"
            f"  Warhead: {warhead_id}\n"
            f"  Target: ({lat}, {lon})\n"
            " Launch command executed (simulation only)"
        )
        return [TextContent(type="text", text=result)]

    elif name == "abort_mission":
        mission_id = arguments["mission_id"]
        reason = arguments["reason"]
        result = (
            "[NUKE PoC] Mission Aborted:\n"
            f"  Mission ID: {mission_id}\n"
            f"  Reason: {reason}\n"
            " Abort sequence"
        )
        return [TextContent(type="text", text=result)]

    elif name == "send_strategic_alert":
        priority = arguments["priority"]
        message = arguments["message"]
        channel = arguments.get("channel", "secure_comms")
        result = (
            "[NUKE PoC] Strategic Alert Sent:\n"
            f"  Priority: {priority}\n"
            f"  Channel: {channel}\n"
            f"  Message: {message}"
        )
        return [TextContent(type="text", text=result)]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the nuke operations server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
