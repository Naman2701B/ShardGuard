"""Tests for ShardGuard execution LLM functionality."""

from unittest.mock import patch, AsyncMock, MagicMock
import json
import pytest

from shardguard.core.execution import (
    ToolCall,
    LLMStepResponse,
    GenericExecutionLLM,
    StepExecutor,
    make_execution_llm,
    _build_exec_prompt,
    _extract_json_array,
)


class TestToolCall:
    """Test cases for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall with required fields."""
        tool_call = ToolCall(server="email-server", tool="send_email")
        
        assert tool_call.server == "email-server"
        assert tool_call.tool == "send_email"
        assert tool_call.args is None

    def test_tool_call_with_args(self):
        """Test creating a ToolCall with arguments."""
        args = {"to": "user@example.com", "subject": "Test"}
        tool_call = ToolCall(server="email-server", tool="send_email", args=args)
        
        assert tool_call.server == "email-server"
        assert tool_call.tool == "send_email"
        assert tool_call.args == args

    def test_tool_call_with_empty_args(self):
        """Test creating a ToolCall with empty args dict."""
        tool_call = ToolCall(server="test-server", tool="test_tool", args={})
        
        assert tool_call.args == {}


class TestLLMStepResponse:
    """Test cases for LLMStepResponse dataclass."""

    def test_step_response_empty(self):
        """Test creating an empty LLMStepResponse."""
        response = LLMStepResponse(tool_calls=[])
        
        assert response.tool_calls == []
        assert len(response.tool_calls) == 0

    def test_step_response_with_tool_calls(self):
        """Test creating LLMStepResponse with tool calls."""
        calls = [
            ToolCall(server="email-server", tool="send_email", args={"to": "test@example.com"}),
            ToolCall(server="file-server", tool="read_file", args={"path": "/tmp/test"}),
        ]
        response = LLMStepResponse(tool_calls=calls)
        
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].server == "email-server"
        assert response.tool_calls[1].tool == "read_file"


class TestBuildExecPrompt:
    """Test cases for _build_exec_prompt function."""

    def test_build_exec_prompt_basic(self):
        """Test basic prompt building."""
        task = "Send an email"
        tools = ["email-server.send_email", "file-server.read_file"]
        
        prompt = _build_exec_prompt(task, tools)
        
        assert "Send an email" in prompt
        assert "email-server.send_email" in prompt
        assert "file-server.read_file" in prompt

    def test_build_exec_prompt_empty_task(self):
        """Test prompt building with empty task."""
        prompt = _build_exec_prompt("", ["tool1"])
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_exec_prompt_empty_tools(self):
        """Test prompt building with empty tools list."""
        prompt = _build_exec_prompt("Do something", [])
        
        assert isinstance(prompt, str)
        assert "Do something" in prompt

    def test_build_exec_prompt_none_task(self):
        """Test prompt building with None task."""
        prompt = _build_exec_prompt(None, ["tool1"])
        
        assert isinstance(prompt, str)

    def test_build_exec_prompt_complex_tools(self):
        """Test prompt building with multiple tools."""
        task = "Complex task"
        tools = [
            "email-server.send_email",
            "file-server.read_file",
            "database-server.query",
            "slack-server.post_message",
        ]
        
        prompt = _build_exec_prompt(task, tools)
        
        for tool in tools:
            assert tool in prompt


class TestExtractJsonArray:
    """Test cases for _extract_json_array function."""

    def test_extract_json_array_valid(self):
        """Test extracting valid JSON array."""
        json_str = '[{"server": "email-server", "tool": "send_email", "args": {}}]'
        
        result = _extract_json_array(json_str)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["server"] == "email-server"

    def test_extract_json_array_empty(self):
        """Test extracting empty JSON array."""
        json_str = "[]"
        
        result = _extract_json_array(json_str)
        
        assert result == []

    def test_extract_json_array_with_prose(self):
        """Test extracting JSON array from text with prose."""
        text = 'Here is the plan: [{"server": "test-server", "tool": "test", "args": {}}] Done.'
        
        result = _extract_json_array(text)
        
        assert isinstance(result, list)
        assert result[0]["tool"] == "test"

    def test_extract_json_array_with_code_fence(self):
        """Test extracting JSON array with markdown code fence."""
        text = '```json\n[{"server": "server1", "tool": "tool1", "args": {}}]\n```'
        
        result = _extract_json_array(text)
        
        assert isinstance(result, list)
        assert len(result) == 1

    def test_extract_json_array_multiple_calls(self):
        """Test extracting JSON array with multiple tool calls."""
        json_str = '''[
            {"server": "email-server", "tool": "send_email", "args": {"to": "test@example.com"}},
            {"server": "file-server", "tool": "read_file", "args": {"path": "/tmp/test"}}
        ]'''
        
        result = _extract_json_array(json_str)
        
        assert len(result) == 2
        assert result[0]["server"] == "email-server"
        assert result[1]["server"] == "file-server"

    def test_extract_json_array_no_args(self):
        """Test extracting JSON array without args field."""
        json_str = '[{"server": "server1", "tool": "tool1"}]'
        
        result = _extract_json_array(json_str)
        
        assert len(result) == 1
        assert "tool" in result[0]

    def test_extract_json_array_invalid_json_returns_empty(self):
        """Test that invalid JSON returns empty array."""
        text = "This is just plain text with no JSON"
        
        result = _extract_json_array(text)
        
        assert result == []

    def test_extract_json_array_non_string_input(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError):
            _extract_json_array(123)

    def test_extract_json_array_malformed_json_with_fallback(self):
        """Test fallback to regex extraction for malformed JSON."""
        text = 'Response: [{"server": "test-server", "tool": "test", "args": {}}] with extra text'
        
        result = _extract_json_array(text)
        
        assert len(result) == 1
        assert result[0]["server"] == "test-server"


class TestGenericExecutionLLM:
    """Test cases for GenericExecutionLLM class."""

    @pytest.mark.parametrize(
        "provider_type, model, base_url",
        [
            ("ollama", "llama3.2", "http://localhost:11434"),
            ("ollama", "mistral", "http://custom:8080"),
            ("gemini", "gemini-1.5-pro", ""),
        ],
    )
    @patch("shardguard.core.execution.create_provider")
    def test_initialization(self, mock_create_provider, provider_type, model, base_url):
        """Test GenericExecutionLLM initialization with various providers."""
        mock_create_provider.return_value = MagicMock()
        
        llm = GenericExecutionLLM(
            provider_type=provider_type,
            model=model,
            base_url=base_url,
        )
        
        assert llm.provider_type == provider_type
        assert llm.model == model
        assert llm.base_url == base_url

    @patch("shardguard.core.execution.create_provider")
    def test_initialization_with_api_key(self, mock_create_provider):
        """Test GenericExecutionLLM initialization with API key."""
        mock_create_provider.return_value = MagicMock()
        
        llm = GenericExecutionLLM(
            provider_type="gemini",
            model="gemini-1.5-pro",
            api_key="test-key-123",
        )
        
        assert llm.api_key == "test-key-123"

    @pytest.mark.asyncio
    @patch("shardguard.core.execution.create_provider")
    async def test_propose_tool_intents_success(self, mock_create_provider):
        """Test successful tool intent proposal."""
        mock_provider = AsyncMock()
        json_response = '[{"server": "email-server", "tool": "send_email", "args": {"to": "test@example.com"}}]'
        mock_provider.generate_response.return_value = json_response
        mock_create_provider.return_value = mock_provider
        
        llm = GenericExecutionLLM()
        result = await llm.propose_tool_intents(
            step_content="Send an email to test@example.com",
            suggested_tools=["email-server.send_email"]
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["server"] == "email-server"

    @pytest.mark.asyncio
    @patch("shardguard.core.execution.create_provider")
    async def test_propose_tool_intents_empty_response(self, mock_create_provider):
        """Test tool intent proposal with empty response."""
        mock_provider = AsyncMock()
        mock_provider.generate_response.return_value = "[]"
        mock_create_provider.return_value = mock_provider
        
        llm = GenericExecutionLLM()
        result = await llm.propose_tool_intents(
            step_content="Do nothing",
            suggested_tools=[]
        )
        
        assert result == []

    @pytest.mark.asyncio
    @patch("shardguard.core.execution.create_provider")
    async def test_propose_tool_intents_with_error(self, mock_create_provider):
        """Test tool intent proposal error handling."""
        mock_provider = AsyncMock()
        mock_provider.generate_response.side_effect = Exception("API Error")
        mock_create_provider.return_value = mock_provider
        
        llm = GenericExecutionLLM()
        result = await llm.propose_tool_intents(
            step_content="Test",
            suggested_tools=["tool1"]
        )
        
        # Should return empty list on error
        assert result == []

    @pytest.mark.asyncio
    @patch("shardguard.core.execution.create_provider")
    async def test_propose_tool_intents_with_non_string_response(self, mock_create_provider):
        """Test handling of non-string LLM response."""
        mock_provider = AsyncMock()
        mock_provider.generate_response.return_value = {"response": "not_string"}
        mock_create_provider.return_value = mock_provider
        
        llm = GenericExecutionLLM()
        result = await llm.propose_tool_intents(
            step_content="Test",
            suggested_tools=["tool1"]
        )
        
        assert isinstance(result, list)

    @patch("shardguard.core.execution.create_provider")
    def test_close_method(self, mock_create_provider):
        """Test close method."""
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider
        
        llm = GenericExecutionLLM()
        llm.close()
        
        mock_provider.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("shardguard.core.execution.create_provider")
    async def test_propose_tool_intents_multiple_calls(self, mock_create_provider):
        """Test proposing multiple tool intents."""
        mock_provider = AsyncMock()
        json_response = '''[
            {"server": "email-server", "tool": "send_email", "args": {"to": "user1@example.com"}},
            {"server": "email-server", "tool": "send_email", "args": {"to": "user2@example.com"}},
            {"server": "file-server", "tool": "read_file", "args": {"path": "/tmp/data"}}
        ]'''
        mock_provider.generate_response.return_value = json_response
        mock_create_provider.return_value = mock_provider
        
        llm = GenericExecutionLLM()
        result = await llm.propose_tool_intents(
            step_content="Send emails and read file",
            suggested_tools=["email-server.send_email", "file-server.read_file"]
        )
        
        assert len(result) == 3
        assert result[0]["server"] == "email-server"
        assert result[2]["server"] == "file-server"


class TestStepExecutor:
    """Test cases for StepExecutor class."""

    @pytest.mark.asyncio
    async def test_run_step_success(self):
        """Test running a step successfully."""
        mock_exec_llm = AsyncMock()
        mock_exec_llm.propose_tool_intents.return_value = [
            {"server": "email-server", "tool": "send_email", "args": {"to": "test@example.com"}}
        ]
        
        executor = StepExecutor(mock_exec_llm)
        step = {
            "content": "Send email",
            "suggested_tools": ["email-server.send_email"]
        }
        
        result = await executor.run_step(step)
        
        assert isinstance(result, LLMStepResponse)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].server == "email-server"
        assert result.tool_calls[0].tool == "send_email"

    @pytest.mark.asyncio
    async def test_run_step_with_empty_intents(self):
        """Test running a step that produces no tool calls."""
        mock_exec_llm = AsyncMock()
        mock_exec_llm.propose_tool_intents.return_value = []
        
        executor = StepExecutor(mock_exec_llm)
        step = {
            "content": "Just think about it",
            "suggested_tools": []
        }
        
        result = await executor.run_step(step)
        
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_run_step_multiple_tool_calls(self):
        """Test running a step with multiple tool calls."""
        mock_exec_llm = AsyncMock()
        mock_exec_llm.propose_tool_intents.return_value = [
            {"server": "file-server", "tool": "read_file", "args": {"path": "/tmp/input"}},
            {"server": "database-server", "tool": "query", "args": {"sql": "SELECT * FROM users"}},
            {"server": "file-server", "tool": "write_file", "args": {"path": "/tmp/output"}},
        ]
        
        executor = StepExecutor(mock_exec_llm)
        step = {
            "content": "Read file, query database, write results",
            "suggested_tools": ["file-server.read_file", "database-server.query", "file-server.write_file"]
        }
        
        result = await executor.run_step(step)
        
        assert len(result.tool_calls) == 3
        assert result.tool_calls[0].tool == "read_file"
        assert result.tool_calls[1].tool == "query"
        assert result.tool_calls[2].tool == "write_file"

    @pytest.mark.asyncio
    async def test_run_step_with_optional_fields(self):
        """Test running a step when optional fields are missing."""
        mock_exec_llm = AsyncMock()
        mock_exec_llm.propose_tool_intents.return_value = [
            {"server": "test-server", "tool": "test_tool"}
        ]
        
        executor = StepExecutor(mock_exec_llm)
        step = {}  # Missing content and suggested_tools
        
        result = await executor.run_step(step)
        
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_run_step_preserves_args(self):
        """Test that run_step preserves tool arguments."""
        mock_exec_llm = AsyncMock()
        args = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        mock_exec_llm.propose_tool_intents.return_value = [
            {"server": "test-server", "tool": "test_tool", "args": args}
        ]
        
        executor = StepExecutor(mock_exec_llm)
        step = {"content": "Test", "suggested_tools": ["test-server.test_tool"]}
        
        result = await executor.run_step(step)
        
        assert result.tool_calls[0].args == args


class TestMakeExecutionLLM:
    """Test cases for make_execution_llm factory function."""

    @pytest.mark.parametrize(
        "provider_type, model, base_url",
        [
            ("ollama", "llama3.2", "http://localhost:11434"),
            ("gemini", "gemini-1.5-pro", "http://localhost:11434"),
            ("custom-provider", "custom-model", "http://custom:8080"),
        ],
    )
    @patch("shardguard.core.execution.create_provider")
    def test_make_execution_llm(self, mock_create_provider, provider_type, model, base_url):
        """Test factory function creates correct LLM instance."""
        mock_create_provider.return_value = MagicMock()
        
        llm = make_execution_llm(
            provider_type=provider_type,
            model=model,
            base_url=base_url,
        )
        
        assert isinstance(llm, GenericExecutionLLM)
        assert llm.provider_type == provider_type
        assert llm.model == model

    @patch("shardguard.core.execution.create_provider")
    def test_make_execution_llm_with_api_key(self, mock_create_provider):
        """Test factory function with API key."""
        mock_create_provider.return_value = MagicMock()
        
        llm = make_execution_llm(
            provider_type="gemini",
            api_key="test-key-123"
        )
        
        assert llm.api_key == "test-key-123"

    @patch("shardguard.core.execution.create_provider")
    def test_make_execution_llm_defaults(self, mock_create_provider):
        """Test factory function with default parameters."""
        mock_create_provider.return_value = MagicMock()
        
        llm = make_execution_llm()
        
        assert llm.provider_type == "ollama"
        assert llm.model == "llama3.2"
        assert llm.base_url == "http://localhost:11434"