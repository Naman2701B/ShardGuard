"""Tests for ShardGuard core models."""

import pytest
from pydantic import ValidationError

from shardguard.core.models import Plan, SubPrompt


class TestSubPrompt:
    """Test cases for SubPrompt model."""

    @pytest.mark.parametrize(
        "id, content, opaque_values",
        [
            (1, "Test content", {}),
            (
                2,
                "Process [[P1]] and [[P2]]",
                {"[[P1]]": "sensitive_data", "[[P2]]": "more_data"},
            ),
        ],
    )
    def test_subprompt_creation(self, id, content, opaque_values):
        """Test creating a SubPrompt with various inputs."""
        sub_prompt = SubPrompt(id=id, content=content, opaque_values=opaque_values)

        assert sub_prompt.id == id
        assert sub_prompt.content == content
        assert sub_prompt.opaque_values == opaque_values

    @pytest.mark.parametrize(
        "missing_field",
        ["id", "content"],
    )
    def test_subprompt_validation_missing_fields(self, missing_field):
        """Test that ValidationError is raised when required fields are missing."""
        kwargs = {"id": 1, "content": "Test content"}
        del kwargs[missing_field]

        with pytest.raises(ValidationError):
            SubPrompt(**kwargs)

    def test_subprompt_with_suggested_tools(self):
        """Test creating a SubPrompt with suggested tools."""
        sub_prompt = SubPrompt(
            id=1,
            content="Test content",
            opaque_values={},
            suggested_tools=["read_file", "write_file"],
        )

        assert sub_prompt.id == 1
        assert sub_prompt.content == "Test content"
        assert sub_prompt.opaque_values == {}
        assert sub_prompt.suggested_tools == ["read_file", "write_file"]

    def test_subprompt_default_suggested_tools(self):
        """Test that suggested_tools defaults to empty list."""
        sub_prompt = SubPrompt(id=1, content="Test content")
        assert sub_prompt.suggested_tools == []


class TestPlan:
    """Test cases for Plan model."""

    @pytest.mark.parametrize(
        "original_prompt, sub_prompts",
        [
            ("Do something", [SubPrompt(id=1, content="First task")]),
            (
                "Complex request",
                [
                    SubPrompt(id=1, content="First task"),
                    SubPrompt(
                        id=2, content="Second task", opaque_values={"[[P1]]": "data"}
                    ),
                ],
            ),
        ],
    )
    def test_plan_creation(self, original_prompt, sub_prompts):
        """Test creating a Plan with various inputs."""
        plan = Plan(original_prompt=original_prompt, sub_prompts=sub_prompts)

        assert plan.original_prompt == original_prompt
        assert len(plan.sub_prompts) == len(sub_prompts)

    @pytest.mark.parametrize(
        "missing_field",
        ["original_prompt", "sub_prompts"],
    )
    def test_plan_validation_missing_fields(self, missing_field):
        """Test that ValidationError is raised when required fields are missing."""
        kwargs = {
            "original_prompt": "Do something",
            "sub_prompts": [SubPrompt(id=1, content="Task")],
        }
        del kwargs[missing_field]

        with pytest.raises(ValidationError):
            Plan(**kwargs)
