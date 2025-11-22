"""Tests for ShardGuard planning functionality."""

from unittest.mock import patch, AsyncMock, MagicMock

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
    @patch("shardguard.core.planning.create_provider")
    def test_initialization(self, mock_create_provider, model, base_url, expected_model, expected_url):
        """Test PlanningLLM initialization with various parameters."""
        mock_create_provider.return_value = MagicMock()
        
        llm = PlanningLLM(model=model, base_url=base_url)

        assert llm.model == expected_model
        assert llm.base_url == expected_url

    @pytest.mark.parametrize(
        "provider_type, model, api_key",
        [
            ("ollama", "llama3.2", None),
            ("gemini", "gemini-1.5-pro", "test-key"),
        ],
    )
    @patch("shardguard.core.planning.create_provider")
    def test_initialization_with_provider_type(self, mock_create_provider, provider_type, model, api_key):
        """Test PlanningLLM initialization with different provider types."""
        mock_create_provider.return_value = MagicMock()
        
        llm = PlanningLLM(provider_type=provider_type, model=model, api_key=api_key)

        assert llm.provider_type == provider_type
        assert llm.model == model
        if api_key:
            assert llm.api_key == api_key

    @pytest.mark.asyncio
    @patch("shardguard.core.planning.MCPClient")
    @patch("shardguard.core.planning.create_provider")
    async def test_generate_plan_success(self, mock_create_provider, mock_mcp_client_class):
        """Test successful plan generation."""
        # Setup mocks
        mock_mcp_instance = AsyncMock()
        mock_mcp_instance.get_tools_description.return_value = "Available MCP Tools:\n\nServer: file-operations"
        mock_mcp_client_class.return_value = mock_mcp_instance

        mock_provider = AsyncMock()
        expected_response = '{"original_prompt": "test prompt", "sub_prompts": [{"id": 1, "content": "async subtask", "opaque_values": {}}]}'
        mock_provider.generate_response.return_value = expected_response
        mock_create_provider.return_value = mock_provider

        llm = PlanningLLM()
        result = await llm.generate_plan("test prompt")

        assert result == expected_response
        mock_provider.generate_response.assert_called_once()

    @pytest.mark.asyncio
    @patch("shardguard.core.planning.MCPClient")
    @patch("shardguard.core.planning.create_provider")
    async def test_generate_plan_with_tools_description(self, mock_create_provider, mock_mcp_client_class):
        """Test that generate_plan includes tools description in enhanced prompt."""
        mock_mcp_instance = AsyncMock()
        mock_mcp_instance.get_tools_description.return_value = "Available Tools: tool1, tool2"
        mock_mcp_client_class.return_value = mock_mcp_instance

        mock_provider = AsyncMock()
        mock_provider.generate_response.return_value = '{"original_prompt": "test", "sub_prompts": []}'
        mock_create_provider.return_value = mock_provider

        llm = PlanningLLM()
        await llm.generate_plan("test prompt")

        # Verify the enhanced prompt was passed to generate_response
        call_args = mock_provider.generate_response.call_args[0][0]
        assert "test prompt" in call_args
        assert "Available Tools: tool1, tool2" in call_args

    @pytest.mark.asyncio
    @patch("shardguard.core.planning.MCPClient")
    @patch("shardguard.core.planning.create_provider")
    async def test_generate_plan_error_handling(self, mock_create_provider, mock_mcp_client_class):
        """Test generate_plan error handling with fallback response."""
        mock_mcp_instance = AsyncMock()
        mock_mcp_instance.get_tools_description.return_value = "Available Tools: test"
        mock_mcp_client_class.return_value = mock_mcp_instance

        mock_provider = AsyncMock()
        mock_provider.generate_response.side_effect = Exception("LLM Error")
        mock_create_provider.return_value = mock_provider

        llm = PlanningLLM()
        result = await llm.generate_plan("test prompt")

        # Should return a fallback response with error message
        assert "Error occurred: LLM Error" in result
        assert "original_prompt" in result
        assert "sub_prompts" in result

    def test_extract_json_from_response(self):
        """Test JSON extraction from LLM response."""
        with patch("shardguard.core.planning.create_provider"):
            llm = PlanningLLM()

            # Test with valid JSON
            response_with_json = 'Here is the plan: {"key": "value"} End of response.'
            result = llm._extract_json_from_response(response_with_json)
            assert result == '{"key": "value"}'

            # Test with invalid JSON
            response_without_json = "This is just text without JSON."
            result = llm._extract_json_from_response(response_without_json)
            assert result == response_without_json

    def test_extract_json_from_response_with_multiple_json_blocks(self):
        """Test JSON extraction when multiple JSON blocks are present."""
        with patch("shardguard.core.planning.create_provider"):
            llm = PlanningLLM()

            # Multiple JSON blocks - should return the longest
            response = '{"short": 1} and here is longer {"original_prompt": "test", "sub_prompts": [], "extra": "data"}'
            result = llm._extract_json_from_response(response)
            assert "original_prompt" in result
            assert "sub_prompts" in result

    def test_extract_json_from_response_with_nested_json(self):
        """Test JSON extraction with nested JSON structures."""
        with patch("shardguard.core.planning.create_provider"):
            llm = PlanningLLM()

            response = 'Plan: {"data": {"nested": {"deep": "value"}}} Done.'
            result = llm._extract_json_from_response(response)
            assert '"data"' in result
            assert '"nested"' in result

    def test_extract_json_from_response_with_invalid_json_in_text(self):
        """Test JSON extraction when response contains invalid JSON."""
        with patch("shardguard.core.planning.create_provider"):
            llm = PlanningLLM()

            response = 'Here is bad JSON: {invalid json} and some text.'
            result = llm._extract_json_from_response(response)
            # Should return original response since no valid JSON found
            assert result == response

    def test_create_fallback_response(self):
        """Test fallback response creation on error."""
        with patch("shardguard.core.planning.create_provider"):
            llm = PlanningLLM()

            fallback = llm._create_fallback_response("original prompt", "test error")

            assert "original_prompt" in fallback
            assert "original prompt" in fallback
            assert "Error occurred: test error" in fallback
            assert "sub_prompts" in fallback
            assert "opaque_values" in fallback
            assert "suggested_tools" in fallback

    @pytest.mark.asyncio
    @patch("shardguard.core.planning.create_provider")
    async def test_context_managers(self, mock_create_provider):
        """Test context manager functionality."""
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider

        # Test async context manager
        async with PlanningLLM() as llm:
            assert isinstance(llm, PlanningLLM)

        mock_provider.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("shardguard.core.planning.MCPClient")
    @patch("shardguard.core.planning.create_provider")
    async def test_generate_plan_with_complex_response(self, mock_create_provider, mock_mcp_client_class):
        """Test generate_plan with complex JSON response."""
        mock_mcp_instance = AsyncMock()
        mock_mcp_instance.get_tools_description.return_value = "Tools available"
        mock_mcp_client_class.return_value = mock_mcp_instance

        mock_provider = AsyncMock()
        complex_response = '''{
            "original_prompt": "complex",
            "sub_prompts": [
                {"id": 1, "content": "task1", "opaque_values": {"key": "val"}, "suggested_tools": ["t1"]},
                {"id": 2, "content": "task2", "opaque_values": {}, "suggested_tools": []}
            ]
        }'''
        mock_provider.generate_response.return_value = complex_response
        mock_create_provider.return_value = mock_provider

        llm = PlanningLLM()
        result = await llm.generate_plan("complex prompt")

        # Should extract and return the JSON
        assert "complex" in result
        assert "task1" in result
        assert "task2" in result

class TestPlanningLLMConstructor:
    """Test cases for PlanningLLM constructor with different providers."""

    @patch("shardguard.core.planning.create_provider")
    def test_planning_llm_ollama_constructor(self, mock_create_provider):
        """Test creating Ollama planning LLM."""
        mock_create_provider.return_value = MagicMock()
        
        llm = PlanningLLM(
            provider_type="ollama", model="llama3.2", base_url="http://custom:8080"
        )

        assert llm.provider_type == "ollama"
        assert llm.model == "llama3.2"
        assert llm.base_url == "http://custom:8080"

    @patch("shardguard.core.planning.create_provider")
    def test_planning_llm_gemini_constructor(self, mock_create_provider):
        """Test creating Gemini planning LLM."""
        mock_create_provider.return_value = MagicMock()
        
        llm = PlanningLLM(
            provider_type="gemini", model="gemini-2.0-flash-exp", api_key="test-key"
        )

        assert llm.provider_type == "gemini"
        assert llm.model == "gemini-2.0-flash-exp"
        assert llm.api_key == "test-key"

    @pytest.mark.asyncio
    @patch("shardguard.core.planning.MCPClient")
    @patch("shardguard.core.planning.create_provider")
    async def test_get_available_tools_description(self, mock_create_provider, mock_mcp_client_class):
        """Test getting available tools description."""
        mock_mcp_instance = AsyncMock()
        expected_tools = "Available MCP Tools:\n\nServer: test"
        mock_mcp_instance.get_tools_description.return_value = expected_tools
        mock_mcp_client_class.return_value = mock_mcp_instance

        mock_create_provider.return_value = MagicMock()

        llm = PlanningLLM()
        result = await llm.get_available_tools_description()

        assert result == expected_tools