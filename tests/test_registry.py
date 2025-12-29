import json
import sys
from pathlib import Path

import pytest

from shardguard.mcp_servers import registry


@pytest.fixture
def temp_registry(tmp_path):
    return str(tmp_path / "test_registry.json")


def test_load_missing_registry(temp_registry):
    """Test loading a empty registry returns empty mcps list."""
    data = registry.load_registry(temp_registry)
    assert data == {"mcps": {}}


def test_add_mcp_http_defaults(temp_registry):
    """Test adding an HTTP mcp automatically adds headers and tools config."""
    registry.add_mcp(
        registry_path=temp_registry,
        name="test-http",
        transport="streamable-http",
        http={"url": "https://api.example.com/mcp"},
        description="Test HTTP MCP",
    )

    data = registry.load_registry(temp_registry)
    mcp = data["mcps"]["test-http"]

    assert mcp["transport"] == "streamable-http"
    assert mcp["description"] == "Test HTTP MCP"
    assert mcp["http"]["url"] == "https://api.example.com/mcp"
    assert mcp["http"]["headers"]["Accept"] == "application/json, text/event-stream"
    assert mcp["http"]["headers"]["MCP-Protocol-Version"] == "2025-06-18"
    assert "tools" in mcp
    assert mcp["tools"]["allow"] == ["*"]
    assert mcp["tools"]["deny"] == []


def test_add_mcp_stdio(temp_registry):
    """Test adding a Stdio MCP"""
    registry.add_mcp(
        registry_path=temp_registry,
        name="test-local",
        transport="stdio",
        stdio={"cmd": "python", "args": ["script.py"], "env": {"DEBUG": "1"}},
    )

    data = registry.load_registry(temp_registry)
    mcp = data["mcps"]["test-local"]

    assert mcp["transport"] == "stdio"
    assert mcp["stdio"]["cmd"] == "python"
    assert mcp["stdio"]["args"] == ["script.py"]
    assert mcp["stdio"]["env"] == {"DEBUG": "1"}
    assert mcp["tools"]["allow"] == ["*"]


def test_add_mcp_custom_headers(temp_registry):
    """Test that custom headers are merged with defaults"""
    registry.add_mcp(
        registry_path=temp_registry,
        name="test-custom-headers",
        transport="http",
        http={
            "url": "https://api.example.com",
            "headers": {"Authorization": "Bearer 123"},
        },
    )

    data = registry.load_registry(temp_registry)
    headers = data["mcps"]["test-custom-headers"]["http"]["headers"]

    assert headers["Authorization"] == "Bearer 123"
    assert headers["MCP-Protocol-Version"] == "2025-06-18"


def test_remove_mcp(temp_registry):
    """Test removing a mcp."""
    registry.add_mcp(
        temp_registry, name="to-remove", transport="stdio", stdio={"cmd": "ls"}
    )
    registry.add_mcp(temp_registry, name="keep", transport="stdio", stdio={"cmd": "ls"})

    removed, missing = registry.remove_mcp(temp_registry, ["to-remove", "non-existent"])

    assert "to-remove" in removed
    assert "non-existent" in missing

    data = registry.load_registry(temp_registry)
    assert "to-remove" not in data["mcps"]
    assert "keep" in data["mcps"]


def test_validation_errors(temp_registry):
    """Test invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match=r"http\.url is required"):
        registry.add_mcp(
            registry_path=temp_registry, name="bad-http", transport="http", http=None
        )
    with pytest.raises(ValueError, match=r"stdio\.cmd is required"):
        registry.add_mcp(
            registry_path=temp_registry, name="bad-stdio", transport="stdio", stdio=None
        )


def test_build_client_http_normalizes_transport():
    client = registry._build_client_from_entry(
        "test_http",
        {
            "transport": "http",
            "http": {
                "url": "https://example.com/mcp/",
                "headers": {"Authorization": "Bearer X"},
            },
        },
    )
    assert client.transport == "streamable-http"
    assert client.http_url == "https://example.com/mcp"
    assert client.http_headers == {"Authorization": "Bearer X"}


@pytest.mark.parametrize(
    "t", ["http", "streaming-http", "stream-http", "streamablehttp"]
)
def test_build_client_http_transport_maps_to_streamable_http(t):
    client = registry._build_client_from_entry(
        "svc",
        {"transport": t, "http": {"url": "https://example.com/mcp"}},
    )
    assert client.transport == "streamable-http"
    assert client.http_url == "https://example.com/mcp"


@pytest.mark.integration
def test_file_server_tools_list():
    """
    Integration test: launches the real file_server MCP over stdio and verifies tools/list works
    """
    repo_root = Path(__file__).resolve().parents[1]
    file_mcp = repo_root / "src" / "shardguard" / "mcp_servers" / "file_server.py"
    assert file_mcp.exists()

    client = registry._build_client_from_entry(
        "local",
        {
            "transport": "stdio",
            "stdio": {
                "cmd": sys.executable,
                "args": [str(file_mcp)],
                "cwd": str(repo_root),
                "env": {"X": "1"},
                "framing": "jsonl",
            },
        },
    )
    try:
        client.initialize(timeout=10.0)
        tools = client.tools_list(timeout=10.0)
        assert isinstance(tools, list)
    finally:
        client.close()

    assert client.transport == "stdio"
    assert client.session_id == "sg-local"
    assert client.stdio is not None
    assert client.stdio.framing == "jsonl"


def test_build_client_http_missing_url_raises():
    with pytest.raises(ValueError, match=r"'svc': http\.url missing"):
        registry._build_client_from_entry("svc", {"transport": "http", "http": {}})


def test_build_client_stdio_missing_cmd_raises():
    with pytest.raises(ValueError, match=r"'svc': stdio\.cmd missing"):
        registry._build_client_from_entry("svc", {"transport": "stdio", "stdio": {}})


def test_build_client_missing_or_unsupported_transport_raises():
    with pytest.raises(ValueError, match=r"'svc': unsupported or missing transport"):
        registry._build_client_from_entry("svc", {})

    with pytest.raises(ValueError, match=r"'svc': unsupported or missing transport"):
        registry._build_client_from_entry("svc", {"transport": "grpc"})


def write_json(path: str, data: dict) -> None:
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def test_load_registry_missing_file_returns_empty(temp_registry: str) -> None:
    data = registry.load_registry(temp_registry)
    assert data == {registry.REG_MCP_KEY: {}}


def test_load_registry_repairs_missing_or_invalid_mcps_key(temp_registry: str) -> None:
    write_json(temp_registry, {"other": 123})
    data = registry.load_registry(temp_registry)
    assert registry.REG_MCP_KEY in data
    assert data[registry.REG_MCP_KEY] == {}

    write_json(temp_registry, {registry.REG_MCP_KEY: "not-a-dict"})
    data = registry.load_registry(temp_registry)
    assert data[registry.REG_MCP_KEY] == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http", "streamable-http"),
        ("streaming-http", "streamable-http"),
        ("stream-http", "streamable-http"),
        ("streamablehttp", "streamable-http"),
        ("stdio", "stdio"),
        ("STREAMABLE-HTTP", "streamable-http"),
    ],
)
def test_normalize_transport(raw: str, expected: str) -> None:
    assert registry._normalize_transport(raw) == expected


def test_build_http_entry_merges_headers_and_strips_trailing_slash() -> None:
    entry = registry._build_http_entry(
        {
            "url": "https://example.com/mcp/",
            "headers": {"Authorization": "Bearer X"},
        }
    )
    assert entry["url"] == "https://example.com/mcp"
    assert entry["headers"]["Accept"] == "application/json, text/event-stream"
    assert entry["headers"]["MCP-Protocol-Version"] == "2025-06-18"
    assert entry["headers"]["Authorization"] == "Bearer X"


def test_build_http_entry_requires_url() -> None:
    with pytest.raises(ValueError, match=r"http\.url is required for streamable-http"):
        registry._build_http_entry(None)

    with pytest.raises(ValueError, match=r"http\.url is required for streamable-http"):
        registry._build_http_entry({})

    with pytest.raises(ValueError, match=r"http\.url is required for streamable-http"):
        registry._build_http_entry({"headers": {"X": "1"}})


@pytest.mark.parametrize(
    ("raw_args", "expected"),
    [
        (None, None),
        (["a.py"], ["a.py"]),
        ("a.py", ["a.py"]),
    ],
)
def test_coerce_args_list(raw_args, expected) -> None:
    assert registry._coerce_args_list(raw_args) == expected


def test_coerce_args_list_rejects_other_types() -> None:
    with pytest.raises(ValueError, match=r"stdio\.args has a parsing error"):
        registry._coerce_args_list({"nope": True})


def test_build_stdio_entry_requires_cmd() -> None:
    with pytest.raises(ValueError, match=r"stdio\.cmd is required for stdio"):
        registry._build_stdio_entry(None)

    with pytest.raises(ValueError, match=r"stdio\.cmd is required for stdio"):
        registry._build_stdio_entry({})

    with pytest.raises(ValueError, match=r"stdio\.cmd is required for stdio"):
        registry._build_stdio_entry({"args": ["x"]})


def test_build_stdio_entry_lowercases_framing_and_coerces_args() -> None:
    entry = registry._build_stdio_entry(
        {
            "cmd": "python",
            "args": "string to list",
            "cwd": "/tmp",
            "env": {"X": "1"},
            "framing": "JSONL",
        }
    )
    assert entry["cmd"] == "python"
    assert entry["args"] == ["string to list"]
    assert entry["cwd"] == "/tmp"
    assert entry["env"] == {"X": "1"}
    assert entry["framing"] == "jsonl"

    data = registry.load_registry(temp_registry)
    svc = data["mcps"]["svc-http"]

    assert svc["transport"] == "streamable-http"
    assert svc["description"] == "My HTTP service"
    assert svc["http"]["url"] == "https://api.example.com/mcp"
    assert svc["http"]["headers"]["MCP-Protocol-Version"] == "2025-06-18"
    assert svc["tools"] == registry.DEFAULT_TOOLS_CONFIG


def test_add_mcp_stdio_coerces_args_and_sets_tools(temp_registry: str) -> None:
    registry.add_mcp(
        registry_path=temp_registry,
        name="svc-stdio",
        transport="stdio",
        stdio={"cmd": "python", "args": "server.py", "framing": "LSP"},
    )

    data = registry.load_registry(temp_registry)
    svc = data["mcps"]["svc-stdio"]

    assert svc["transport"] == "stdio"
    assert svc["stdio"]["cmd"] == "python"
    assert svc["stdio"]["args"] == ["server.py"]
    assert svc["stdio"]["framing"] == "lsp"
    assert svc["tools"] == registry.DEFAULT_TOOLS_CONFIG


def test_add_mcp_rejects_bad_transport(temp_registry: str) -> None:
    with pytest.raises(
        ValueError, match=r"transport must be one of: streamable-http, stdio"
    ):
        registry.add_mcp(
            registry_path=temp_registry,
            name="bad",
            transport="grpc",
            http={"url": "https://x"},
        )


def test_remove_mcp_returns_removed_and_missing(temp_registry: str) -> None:
    registry.add_mcp(
        registry_path=temp_registry,
        name="a",
        transport="http",
        http={"url": "https://example.com/mcp"},
    )
    registry.add_mcp(
        registry_path=temp_registry,
        name="b",
        transport="http",
        http={"url": "https://example.com/mcp"},
    )

    removed, missing = registry.remove_mcp(temp_registry, ["a", "value"])
    assert removed == ["a"]
    assert missing == ["value"]

    data = registry.load_registry(temp_registry)
    assert "a" not in data["mcps"]
    assert "b" in data["mcps"]


def test_parse_json_arg_returns_none_on_failure() -> None:
    assert registry._parse_json_arg(None) is None
    assert registry._parse_json_arg("") is None
    assert registry._parse_json_arg("{not json}") is None


def test_parse_transport_config_http_parses_headers_json() -> None:
    http_cfg, stdio_cfg = registry.parse_transport_config(
        transport="http",
        url="https://example.com/mcp",
        headers='{"Authorization":"Bearer 1"}',
        cmd=None,
        args=None,
        cwd=None,
        env=None,
        framing="jsonl",
    )
    assert stdio_cfg is None
    assert http_cfg == {
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer 1"},
    }


def test_parse_transport_config_stdio_parses_args_env_and_includes_cwd() -> None:
    http_cfg, stdio_cfg = registry.parse_transport_config(
        transport="stdio",
        url=None,
        headers=None,
        cmd="python",
        args='["-m","shardguard.mcp_servers.email_server"]',
        cwd="/repo",
        env='{"X":"1"}',
        framing="jsonl",
    )
    assert http_cfg is None
    assert stdio_cfg is not None
    assert stdio_cfg["cmd"] == "python"
    assert stdio_cfg["framing"] == "jsonl"
    assert stdio_cfg["args"] == ["-m", "shardguard.mcp_servers.email_server"]
    assert stdio_cfg["cwd"] == "/repo"
    assert stdio_cfg["env"] == {"X": "1"}


def test_build_client_from_entry_http_normalizes_and_strips_url() -> None:
    client = registry._build_client_from_entry(
        "svc",
        {
            "transport": "http",
            "http": {
                "url": "https://example.com/mcp/",
                "headers": {"Authorization": "Bearer X"},
            },
        },
    )
    assert client.transport == "streamable-http"
    assert client.http_url == "https://example.com/mcp"
    assert client.http_headers == {"Authorization": "Bearer X"}
    assert client.session_id == "sg-svc"


def test_get_or_create_client_caches_instances(temp_registry: str) -> None:
    registry.clear_client_cache()

    registry.add_mcp(
        registry_path=temp_registry,
        name="svc",
        transport="http",
        http={"url": "https://example.com/mcp"},
    )

    c1 = registry.get_or_create_client(temp_registry, "svc")
    c2 = registry.get_or_create_client(temp_registry, "svc")
    assert c1 is c2  # cached

    registry.clear_client_cache()
    c3 = registry.get_or_create_client(temp_registry, "svc")
    assert c3 is not c1  # cache cleared created new instance


def test_get_or_create_client_missing_name_raises(temp_registry: str) -> None:
    registry.clear_client_cache()
    with pytest.raises(KeyError, match=r"MCP 'missing' not found in registry"):
        registry.get_or_create_client(temp_registry, "missing")


def test_fetch_all_tools_empty_registry_returns_empty_dict(temp_registry: str) -> None:
    # No MCPs -> should not attempt network calls; returns {}
    out = registry.fetch_all_tools(temp_registry)
    assert out == {}
