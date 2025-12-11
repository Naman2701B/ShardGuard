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
