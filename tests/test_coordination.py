"""Tests for ShardGuard coordination functionality."""

from unittest.mock import AsyncMock, Mock, patch

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

    def test_format_prompt_method(self):
        """Test the _format_prompt method."""
        mock_planner = MockPlanningLLM()

        with patch("builtins.open", Mock()):
            service = CoordinationService(mock_planner)

            formatted = service._format_prompt("test input")

            # Should contain the input and be based on PLANNING_PROMPT
            assert "test input" in formatted
            assert len(formatted) > len(
                "test input"
            )  # Should be more than just the input

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
