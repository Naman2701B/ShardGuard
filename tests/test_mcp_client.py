import json
from unittest.mock import ANY, MagicMock, patch

import pytest

from shardguard.core.mcp_client import MCPClient, _headers


def test_headers_generation():
    """Verify headers are constructed correctly."""
    h = _headers("sess-123", extra={"Authorization": "Bearer token"})
    assert h["x-session-id"] == "sess-123"
    assert h["Accept"] == "application/json, text/event-stream"
    assert h["Authorization"] == "Bearer token"


@patch("shardguard.core.mcp_client.requests.post")
def test_http_initialize_success(mock_post):
    """Test that initialize handshake works over HTTP."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {"protocolVersion": "2025-06-18"},
    }
    mock_post.return_value = mock_response

    client = MCPClient(
        transport="streamable-http",
        http_url="http://test.remote/mcp",
        session_id="test-session-id",
    )

    result = client.initialize()

    assert result["protocolVersion"] == "2025-06-18"

    mock_post.assert_called()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://test.remote/mcp"
    assert kwargs["headers"]["x-session-id"] == "test-session-id"

    payload = json.loads(kwargs["data"])
    assert payload["method"] == "initialize"
    assert payload["params"]["clientInfo"]["name"] == "shardguard-cli"


@patch("shardguard.core.mcp_client.requests.post")
def test_http_tool_call(mock_post):
    """Test calling a tool over HTTP."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": {"content": [{"type": "text", "text": "success"}]},
    }
    mock_post.return_value = mock_response

    client = MCPClient(transport="streamable-http", http_url="http://test.remote")
    result = client.tools_call("echo", {"msg": "hello"})

    assert len(result) == 1
    assert result[0]["text"] == "success"

    payload = json.loads(mock_post.call_args[1]["data"])
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "echo"
    assert payload["params"]["arguments"] == {"msg": "hello"}


@patch("shardguard.core.mcp_client.requests.post")
def test_http_error_handling(mock_post):
    """Test JSON-RPC errors are raised as exceptions."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": "Method not found"},
    }
    mock_post.return_value = mock_response

    client = MCPClient(transport="streamable-http", http_url="http://test.remote")

    with pytest.raises(RuntimeError, match="Method not found"):
        client.tools_list()


@patch("shardguard.core.mcp_client._StdioRPC")
def test_stdio_initialization(mock_rpc_cls):
    """Test initializing Stdio client."""
    mock_instance = mock_rpc_cls.return_value

    mock_instance.request.return_value = {"serverInfo": {"name": "test-server"}}

    client = MCPClient(
        transport="stdio",
        stdio_cmd="python",
        stdio_args=["server.py"],
        stdio_env={"DEBUG": "1"},
    )

    mock_rpc_cls.assert_called_with(
        "python", ["server.py"], ANY, {"DEBUG": "1"}, framing="jsonl"
    )

    result = client.initialize()
    assert result["serverInfo"]["name"] == "test-server"

    mock_instance.request.assert_called_with("initialize", ANY, timeout=15.0)


@patch("shardguard.core.mcp_client._StdioRPC")
def test_stdio_tool_call(mock_rpc_cls):
    """Test calling tools via Stdio."""
    mock_instance = mock_rpc_cls.return_value
    mock_instance.request.return_value = {
        "content": [{"type": "text", "text": "local-result"}]
    }

    client = MCPClient(transport="stdio", stdio_cmd="cat")
    result = client.tools_call("read_file", {"path": "foo.txt"})

    assert result[0]["text"] == "local-result"

    mock_instance.request.assert_called_with(
        "tools/call",
        {"name": "read_file", "arguments": {"path": "foo.txt"}},
        timeout=60.0,
    )


def test_invalid_transport():
    with pytest.raises(ValueError, match="Unsupported transport"):
        MCPClient(transport="ftp")


def test_http_missing_url():
    with pytest.raises(ValueError, match="requires http_url"):
        MCPClient(transport="streamable-http", http_url=None)


def test_stdio_missing_cmd():
    with pytest.raises(ValueError, match="requires stdio_cmd"):
        MCPClient(transport="stdio", stdio_cmd=None)
