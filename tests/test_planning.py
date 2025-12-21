"""Tests for ShardGuard planning functionality."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from shardguard.core.planning import PlanningLLM


class MockPlanningLLM:
    """Mock implementation of PlanningLLMProtocol for testing."""

    def __init__(self, response: str | None = None):
        self.response = response or '{"original_prompt": "test", "sub_prompts": []}'

    async def generate_plan(self, prompt: str) -> str:
        return self.response


class TestPlanningLLMProtocol:
    """Test cases for PlanningLLMProtocol."""

    @pytest.mark.asyncio
    async def test_protocol_async_implementation(self):
        """Test that MockPlanningLLM implements the async protocol."""
        mock_llm = MockPlanningLLM()

        # Should be able to call generate_plan
        result = await mock_llm.generate_plan("test prompt")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_protocol_with_custom_response(self):
        """Test protocol implementation with custom response."""
        custom_response = '{"original_prompt": "custom", "sub_prompts": [{"id": 1, "content": "task"}]}'
        mock_llm = MockPlanningLLM(custom_response)

        result = await mock_llm.generate_plan("test")
        assert result == custom_response


class TestPlanningLLM:
    """Test cases for PlanningLLM class."""

    @pytest.mark.parametrize(
        "model, base_url, expected_model, expected_url",
        [
            (
                "llama3.2",
                "http://localhost:11434",
                "llama3.2",
                "http://localhost:11434",
            ),
            (
                "custom-model",
                "http://custom:8080",
                "custom-model",
                "http://custom:8080",
            ),
        ],
    )
    def test_initialization(self, model, base_url, expected_model, expected_url):
        """Test PlanningLLM initialization with various parameters."""
        llm = PlanningLLM(model=model, base_url=base_url)

        assert llm.model == expected_model
        assert llm.base_url == expected_url

    @pytest.mark.asyncio
    @patch("shardguard.core.llm_providers.OllamaProvider.generate_response")
    @patch("shardguard.core.mcp_integration.MCPClient.get_tools_description")
    async def test_generate_plan_success(self, mock_get_tools, mock_generate):
        """Test successful plan generation."""
        # Mock tools description
        mock_get_tools.return_value = "Available MCP Tools:\n\nServer: file-operations"

        # Mock LLM response
        expected_response = '{"original_prompt": "test prompt", "sub_prompts": [{"id": 1, "content": "async subtask", "opaque_values": {}}]}'
        mock_generate.return_value = expected_response

        llm = PlanningLLM()
        result = await llm.generate_plan("test prompt")

        assert result == expected_response
        mock_generate.assert_called_once()

    def test_extract_json_from_response(self):
        """Test JSON extraction from LLM response."""
        llm = PlanningLLM()

        # Test with valid JSON
        response_with_json = 'Here is the plan: {"key": "value"} End of response.'
        result = llm._extract_json_from_response(response_with_json)
        assert result == '{"key": "value"}'

        # Test with invalid JSON
        response_without_json = "This is just text without JSON."
        result = llm._extract_json_from_response(response_without_json)
        assert result == response_without_json

    @pytest.mark.asyncio
    async def test_context_managers(self):
        """Test context manager functionality."""
        # Test async context manager
        async with PlanningLLM() as llm:
            assert isinstance(llm, PlanningLLM)


class TestPlanningLLMConstructor:
    """Test cases for PlanningLLM constructor with different providers."""

    def test_planning_llm_ollama_constructor(self):
        """Test creating Ollama planning LLM."""

        llm = PlanningLLM(
            provider_type="ollama", model="llama3.1", base_url="http://custom:8080"
        )

        assert llm.provider_type == "ollama"
        assert llm.model == "llama3.1"
        assert llm.base_url == "http://custom:8080"

    def test_planning_llm_gemini_constructor(self):
        """Test creating Gemini planning LLM."""

        llm = PlanningLLM(
            provider_type="gemini", model="gemini-1.5-pro", api_key="test-key"
        )

        assert llm.provider_type == "gemini"
        assert llm.model == "gemini-1.5-pro"
        assert llm.api_key == "test-key"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_available_tools_description_with_local_mcp(tmp_path):
    registry_path = tmp_path / "mcp_registry.json"

    repo_root = Path(__file__).resolve().parents[1]
    file_mcp = repo_root / "src/shardguard/mcp_servers/file_server.py"
    assert file_mcp.exists()

    registry_path.write_text(
        json.dumps(
            {
                "mcps": {
                    "mcp-name": {
                        "transport": "stdio",
                        "stdio": {
                            "cmd": "python",
                            "args": [str(file_mcp)],
                            "cwd": str(repo_root),
                            "framing": "jsonl",
                        },
                        "tools": {"allow": ["*"], "deny": []},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    planning_obj = PlanningLLM.__new__(PlanningLLM)
    planning_obj.registry_path = str(tmp_path / "mcp_registry.json")

    out = await planning_obj.get_available_tools_description()

    assert "MCP_SERVER: mcp-name" in out
    assert "TOOL_KEY:" in out
    # Testing tools in file_server.py
    assert "write_file" in out
    assert "read_file" in out
