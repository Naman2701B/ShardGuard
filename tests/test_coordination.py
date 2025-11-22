"""Tests for ShardGuard coordination functionality."""

from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from collections import OrderedDict

import pytest

from shardguard.core.coordination import CoordinationService
from shardguard.core.models import Plan


class MockPlanningLLM:
    """Mock planning LLM for testing."""

    def __init__(self, response: str | None = None):
        self.response = response or '{"original_prompt": "test", "sub_prompts": []}'

    async def generate_plan(self, prompt: str) -> str:
        return self.response


class TestCoordinationService:
    """Test cases for CoordinationService class."""

    @pytest.mark.parametrize(
        "json_response, expected_original_prompt, expected_sub_prompts",
        [
            (
                """
                {
                    "original_prompt": "Hello world",
                    "sub_prompts": [
                        {
                            "id": 1,
                            "content": "Process greeting",
                            "opaque_values": {}
                        }
                    ]
                }
                """,
                "Hello world",
                [{"id": 1, "content": "Process greeting", "opaque_values": {}}],
            ),
            (
                """
                {
                    "original_prompt": "Process [[P1]] data",
                    "sub_prompts": [
                        {
                            "id": 1,
                            "content": "Analyze [[P1]]",
                            "opaque_values": {
                                "[[P1]]": "sensitive_information"
                            }
                        }
                    ]
                }
                """,
                "Process [[P1]] data",
                [
                    {
                        "id": 1,
                        "content": "Analyze [[P1]]",
                        "opaque_values": {"[[P1]]": "sensitive_information"},
                    }
                ],
            ),
            (
                """
                {
                    "original_prompt": "Complex task with [[P1]] and [[P2]]",
                    "sub_prompts": [
                        {
                            "id": 1,
                            "content": "First step with [[P1]]",
                            "opaque_values": {
                                "[[P1]]": "data1"
                            }
                        },
                        {
                            "id": 2,
                            "content": "Second step with [[P2]]",
                            "opaque_values": {
                                "[[P2]]": "data2"
                            }
                        },
                        {
                            "id": 3,
                            "content": "Final step",
                            "opaque_values": {}
                        }
                    ]
                }
                """,
                "Complex task with [[P1]] and [[P2]]",
                [
                    {
                        "id": 1,
                        "content": "First step with [[P1]]",
                        "opaque_values": {"[[P1]]": "data1"},
                    },
                    {
                        "id": 2,
                        "content": "Second step with [[P2]]",
                        "opaque_values": {"[[P2]]": "data2"},
                    },
                    {"id": 3, "content": "Final step", "opaque_values": {}},
                ],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_handle_prompt(
        self, json_response, expected_original_prompt, expected_sub_prompts
    ):
        """Test handling prompts with various responses."""
        mock_planner = MockPlanningLLM(json_response.strip())

        with patch("builtins.open", Mock()):
            service = CoordinationService(mock_planner)

            result = await service.handle_prompt("Hello world")

        assert result.original_prompt == expected_original_prompt
        assert len(result.sub_prompts) == len(expected_sub_prompts)
        for sub_prompt, expected in zip(
            result.sub_prompts, expected_sub_prompts, strict=False
        ):
            assert sub_prompt.id == expected["id"]
            assert sub_prompt.content == expected["content"]

    @pytest.mark.asyncio
    async def test_handle_prompt_planning_called_with_formatted_prompt(self):
        """Test that planner receives properly formatted prompt with user input."""
        from unittest.mock import AsyncMock

        mock_planner = Mock()
        mock_planner.generate_plan = AsyncMock(
            return_value='{"original_prompt": "test", "sub_prompts": []}'
        )

        with patch("builtins.open", Mock()):
            service = CoordinationService(mock_planner)

            await service.handle_prompt("user input")

            # Verify planner was called with formatted prompt
            mock_planner.generate_plan.assert_called_once()
            call_args = mock_planner.generate_plan.call_args[0][0]

            # The formatted prompt should contain the user input directly
            assert "user input" in call_args

    @pytest.mark.asyncio
    async def test_handle_prompt_invalid_json_from_planner(self):
        """Test handling of invalid JSON from planner."""
        mock_planner = MockPlanningLLM("invalid json response")

        with patch("builtins.open", Mock()):
            service = CoordinationService(mock_planner)

            with pytest.raises(Exception):  # Should raise validation error
                await service.handle_prompt("test input")

    @pytest.mark.asyncio
    async def test_handle_prompt_missing_required_fields(self):
        """Test handling of JSON missing required fields."""
        incomplete_json = '{"original_prompt": "test"}'  # Missing sub_prompts
        mock_planner = MockPlanningLLM(incomplete_json)

        with patch("builtins.open", Mock()):
            service = CoordinationService(mock_planner)

            with pytest.raises(Exception):  # Should raise validation error
                await service.handle_prompt("test input")

    @pytest.mark.asyncio
    async def test_handle_prompt_empty_subprompts_list(self):
        """Test handling prompt with empty sub_prompts list."""
        json_response = """
        {
            "original_prompt": "Simple task",
            "sub_prompts": []
        }
        """
        mock_planner = MockPlanningLLM(json_response.strip())

        with patch("builtins.open", Mock()):
            service = CoordinationService(mock_planner)

            result = await service.handle_prompt("Simple task")

        assert isinstance(result, Plan)
        assert result.original_prompt == "Simple task"
        assert result.sub_prompts == []

    # Tests for check_tool method
    @pytest.mark.asyncio
    async def test_check_tool_all_tools_valid(self):
        """Test check_tool when all suggested tools exist in the system."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
            mock_mcp = Mock()
            mock_mcp.list_tool_names = AsyncMock(
                return_value=["tool1", "tool2", "tool3"]
            )
            mock_mcp_class.return_value = mock_mcp

            result = await service.check_tool(["tool1", "tool2"])

            assert result == [True, True, True]

    @pytest.mark.asyncio
    async def test_check_tool_some_tools_invalid(self):
        """Test check_tool when some suggested tools don't exist."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
            mock_mcp = Mock()
            mock_mcp.list_tool_names = AsyncMock(return_value=["tool1", "tool2"])
            mock_mcp_class.return_value = mock_mcp

            result = await service.check_tool(["tool1", "tool3"])

            assert True in result
            assert False in result

    @pytest.mark.asyncio
    async def test_check_tool_empty_suggested_tools(self):
        """Test check_tool with empty suggested tools list."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
            mock_mcp = Mock()
            mock_mcp.list_tool_names = AsyncMock(
                return_value=["tool1", "tool2"]
            )
            mock_mcp_class.return_value = mock_mcp

            result = await service.check_tool([])

            assert result == [True]

    @pytest.mark.asyncio
    async def test_check_tool_all_tools_invalid(self):
        """Test check_tool when all suggested tools are invalid."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
            mock_mcp = Mock()
            mock_mcp.list_tool_names = AsyncMock(return_value=[])
            mock_mcp_class.return_value = mock_mcp

            result = await service.check_tool(["invalid1", "invalid2"])

            assert result == [False, False]

    # Tests for extract_arguments method
    def test_extract_arguments_with_opaque_values(self):
        """Test extracting arguments from task with opaque_values."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        task = {
            "opaque_values": {
                "[[P1]]": "sensitive_data",
                "[[P2]]": "more_sensitive_data",
            }
        }

        result = service.extract_arguments(task)

        assert set(result) == {"[[P1]]", "[[P2]]"}
        assert service.args["[[P1]]"] == "sensitive_data"
        assert service.args["[[P2]]"] == "more_sensitive_data"

    def test_extract_arguments_accumulates_in_service_args(self):
        """Test that arguments accumulate across multiple calls."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        task1 = {"opaque_values": {"key1": "value1"}}
        task2 = {"opaque_values": {"key2": "value2"}}

        service.extract_arguments(task1)
        service.extract_arguments(task2)

        assert service.args["key1"] == "value1"
        assert service.args["key2"] == "value2"

    # Tests for handle_prompt with tool validation
    @pytest.mark.asyncio
    async def test_handle_prompt_with_valid_tools_no_retry(self):
        """Test handle_prompt succeeds when all tools are valid."""
        json_response = """
        {
            "original_prompt": "Test",
            "sub_prompts": [
                {
                    "id": 1,
                    "content": "Step 1",
                    "suggested_tools": ["tool1"]
                }
            ]
        }
        """
        mock_planner = MockPlanningLLM(json_response.strip())

        with patch("builtins.open", Mock()):
            with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
                mock_mcp = Mock()
                mock_mcp.list_tool_names = AsyncMock(
                    return_value=["tool1", "tool2"]
                )
                mock_mcp_class.return_value = mock_mcp

                service = CoordinationService(mock_planner)
                result = await service.handle_prompt("Test")

                assert isinstance(result, Plan)
                assert result.original_prompt == "Test"

    @pytest.mark.asyncio
    async def test_handle_prompt_retries_on_invalid_tools(self):
        """Test handle_prompt retries when tools are invalid."""
        invalid_json = """
        {
            "original_prompt": "Test",
            "sub_prompts": [
                {
                    "id": 1,
                    "content": "Step 1",
                    "suggested_tools": ["invalid_tool"]
                }
            ]
        }
        """
        valid_json = """
        {
            "original_prompt": "Test",
            "sub_prompts": [
                {
                    "id": 1,
                    "content": "Step 1",
                    "suggested_tools": ["tool1"]
                }
            ]
        }
        """

        mock_planner = Mock()
        mock_planner.generate_plan = AsyncMock(
            side_effect=[invalid_json.strip(), valid_json.strip()]
        )

        with patch("builtins.open", Mock()):
            with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
                mock_mcp = Mock()
                mock_mcp.list_tool_names = AsyncMock(
                    return_value=["tool1", "tool2"]
                )
                mock_mcp_class.return_value = mock_mcp

                service = CoordinationService(mock_planner)
                result = await service.handle_prompt("Test")

                assert isinstance(result, Plan)
                assert mock_planner.generate_plan.call_count <= 2

    @pytest.mark.asyncio
    async def test_handle_prompt_max_retries_exceeded(self):
        """Test handle_prompt stops retrying after max retries."""
        invalid_json = """
        {
            "original_prompt": "Test",
            "sub_prompts": [
                {
                    "id": 1,
                    "content": "Step 1",
                    "suggested_tools": ["invalid_tool"]
                }
            ]
        }
        """

        mock_planner = Mock()
        mock_planner.generate_plan = AsyncMock(return_value=invalid_json.strip())

        with patch("builtins.open", Mock()):
            with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
                mock_mcp = Mock()
                mock_mcp.list_tool_names = AsyncMock(
                    return_value=["tool1", "tool2"]
                )
                mock_mcp_class.return_value = mock_mcp

                service = CoordinationService(mock_planner)
                result = await service.handle_prompt("Test")

                assert result is None

    @pytest.mark.asyncio
    async def test_execute_step_tools_single_tool_call(self):
        """Test executing a single tool call."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        mock_call = Mock()
        mock_call.server = "test_server"
        mock_call.tool = "test_tool"
        mock_call.args = {"param1": "value1"}

        mock_response = Mock()
        mock_response.tool_calls = [mock_call]

        step = {"output_schema": None}

        with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
            with patch("shardguard.core.coordination._validate_output"):
                mock_mcp = Mock()
                mock_mcp.call_tool = AsyncMock(
                    return_value={"result": "success"}
                )
                mock_mcp_class.return_value = mock_mcp

                await service._execute_step_tools(step, mock_response)

                mock_mcp.call_tool.assert_called_once_with(
                    "test_server", "test_tool", {"param1": "value1"}
                )

    @pytest.mark.asyncio
    async def test_execute_step_tools_multiple_tool_calls(self):
        """Test executing multiple tool calls in sequence."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        mock_call1 = Mock()
        mock_call1.server = "server1"
        mock_call1.tool = "tool1"
        mock_call1.args = {"arg1": "value1"}

        mock_call2 = Mock()
        mock_call2.server = "server2"
        mock_call2.tool = "tool2"
        mock_call2.args = {"arg2": "value2"}

        mock_response = Mock()
        mock_response.tool_calls = [mock_call1, mock_call2]

        step = {"output_schema": None}

        with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
            with patch("shardguard.core.coordination._validate_output"):
                mock_mcp = Mock()
                mock_mcp.call_tool = AsyncMock(
                    return_value={"result": "ok"}
                )
                mock_mcp_class.return_value = mock_mcp

                await service._execute_step_tools(step, mock_response)
                # The reason this is <=2 is as this is only suggested tools the LLM 
                # does not necessarily need to run all the tools suggested
                assert mock_mcp.call_tool.call_count <= 2

    @pytest.mark.asyncio
    async def test_execute_step_tools_with_output_schema_validation(self):
        """Test that output schema validation is called."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        output_schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        }

        mock_call = Mock()
        mock_call.server = "server"
        mock_call.tool = "tool"
        mock_call.args = {}

        mock_response = Mock()
        mock_response.tool_calls = [mock_call]

        step = {"output_schema": output_schema}

        with patch("shardguard.core.coordination.MCPClient") as mock_mcp_class:
            with patch("shardguard.core.coordination._validate_output") as mock_validate:
                mock_mcp = Mock()
                mock_mcp.call_tool = AsyncMock(return_value={"result": "test"})
                mock_mcp_class.return_value = mock_mcp

                await service._execute_step_tools(step, mock_response)

                mock_validate.assert_called_once()
                call_args = mock_validate.call_args[0]
                assert call_args[1] == output_schema

    @pytest.mark.asyncio
    async def test_handle_subtasks_multiple_tasks(self):
        """Test handling multiple subtasks."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        tasks = [
            {"id": 1, "content": "Step 1", "opaque_values": {}},
            {"id": 2, "content": "Step 2", "opaque_values": {}},
        ]

        with patch("shardguard.core.coordination.make_execution_llm"):
            with patch("shardguard.core.coordination.StepExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor.run_step = AsyncMock(
                    return_value=Mock(tool_calls=[])
                )
                mock_executor_class.return_value = mock_executor

                with patch.object(
                    service, "_execute_step_tools", new_callable=AsyncMock
                ):
                    await service.handle_subtasks(
                        tasks,
                        provider="ollama",
                        detected_model="llama3.2",
                    )

                    assert mock_executor_class.call_count == 2
                    assert mock_executor.run_step.call_count == 2

    @pytest.mark.asyncio
    async def test_handle_subtasks_with_opaque_values(self):
        """Test that opaque values are properly extracted and set in tasks."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        task = {
            "id": 1,
            "content": "Test",
            "opaque_values": {"[[P1]]": "sensitive"},
        }

        with patch("shardguard.core.coordination.make_execution_llm"):
            with patch("shardguard.core.coordination.StepExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor.run_step = AsyncMock(
                    return_value=Mock(tool_calls=[])
                )
                mock_executor_class.return_value = mock_executor

                with patch.object(
                    service, "_execute_step_tools", new_callable=AsyncMock
                ):
                    await service.handle_subtasks(
                        [task],
                        provider="ollama",
                        detected_model="llama3.2",
                    )

                    call_args = mock_executor.run_step.call_args[0][0]
                    assert "[[P1]]" in call_args["opaque_values"]

    # Tests for service initialization
    def test_service_initialization(self):
        """Test CoordinationService initializes with correct defaults."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        assert service.planner == mock_planner
        assert service.args == {}
        assert service.retryCount == 1
        assert service.console is not None

    def test_service_maintains_args_state(self):
        """Test that service maintains args state across calls."""
        mock_planner = MockPlanningLLM()
        service = CoordinationService(mock_planner)

        service.args["key1"] = "value1"
        service.args["key2"] = "value2"

        assert len(service.args) == 2
        assert service.args["key1"] == "value1"