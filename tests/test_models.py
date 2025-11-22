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

    def test_subprompt_default_opaque_values(self):
        """Test that opaque_values defaults to empty dict."""
        sub_prompt = SubPrompt(id=1, content="Test content")
        assert sub_prompt.opaque_values == {}
        assert isinstance(sub_prompt.opaque_values, dict)

    def test_subprompt_with_all_fields(self):
        """Test creating a SubPrompt with all fields explicitly set."""
        sub_prompt = SubPrompt(
            id=42,
            content="Complete task",
            opaque_values={"key1": "value1", "key2": "value2"},
            suggested_tools=["tool1", "tool2", "tool3"],
        )
        assert sub_prompt.id == 42
        assert sub_prompt.content == "Complete task"
        assert len(sub_prompt.opaque_values) == 2
        assert len(sub_prompt.suggested_tools) == 3

    def test_subprompt_invalid_id_type(self):
        """Test that non-integer id raises ValidationError."""
        with pytest.raises(ValidationError):
            SubPrompt(id="not_an_int", content="Test content")

    def test_subprompt_invalid_content_type(self):
        """Test that non-string content raises ValidationError."""
        with pytest.raises(ValidationError):
            SubPrompt(id=1, content=123)

    def test_subprompt_invalid_opaque_values_type(self):
        """Test that non-dict opaque_values raises ValidationError."""
        with pytest.raises(ValidationError):
            SubPrompt(id=1, content="Test", opaque_values="not_a_dict")

    def test_subprompt_invalid_suggested_tools_type(self):
        """Test that non-list suggested_tools raises ValidationError."""
        with pytest.raises(ValidationError):
            SubPrompt(id=1, content="Test", suggested_tools="not_a_list")

    def test_subprompt_suggested_tools_non_string_elements(self):
        """Test that suggested_tools list must contain strings."""
        with pytest.raises(ValidationError):
            SubPrompt(id=1, content="Test", suggested_tools=[1, 2, 3])

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

    def test_plan_with_many_sub_prompts(self):
        """Test creating a Plan with many sub-prompts."""
        sub_prompts = [
            SubPrompt(id=i, content=f"Task {i}") for i in range(50)
        ]
        plan = Plan(original_prompt="Complex plan", sub_prompts=sub_prompts)
        assert len(plan.sub_prompts) == 50
        assert plan.sub_prompts == sub_prompts

    def test_plan_with_complex_sub_prompts(self):
        """Test Plan with SubPrompts containing all optional fields."""
        sub_prompts = [
            SubPrompt(
                id=1,
                content="Task 1",
                opaque_values={"[[P1]]": "secret1"},
                suggested_tools=["tool1", "tool2"]
            ),
            SubPrompt(
                id=2,
                content="Task 2",
                opaque_values={"[[P2]]": "secret2", "[[P3]]": "secret3"},
                suggested_tools=["tool3"]
            ),
        ]
        plan = Plan(original_prompt="Complex", sub_prompts=sub_prompts)
        assert len(plan.sub_prompts) == 2
        assert plan.sub_prompts[0].suggested_tools == ["tool1", "tool2"]
        assert len(plan.sub_prompts[1].opaque_values) == 2

    def test_plan_invalid_original_prompt_type(self):
        """Test that non-string original_prompt raises ValidationError."""
        with pytest.raises(ValidationError):
            Plan(
                original_prompt=123,
                sub_prompts=[SubPrompt(id=1, content="Task")]
            )

    def test_plan_invalid_sub_prompts_type(self):
        """Test that non-list sub_prompts raises ValidationError."""
        with pytest.raises(ValidationError):
            Plan(
                original_prompt="Do something",
                sub_prompts="not_a_list"
            )

    def test_plan_sub_prompts_invalid_elements(self):
        """Test that sub_prompts list must contain valid SubPrompt-like objects."""
        with pytest.raises(ValidationError):
            Plan(
                original_prompt="Do something",
                sub_prompts=["invalid_string"]
            )

    def test_plan_model_dump(self):
        """Test that Plan can be serialized to dict."""
        sub_prompt = SubPrompt(
            id=1,
            content="Task",
            opaque_values={"key": "value"},
            suggested_tools=["tool1"]
        )
        plan = Plan(original_prompt="Do something", sub_prompts=[sub_prompt])
        plan_dict = plan.model_dump()
        assert isinstance(plan_dict, dict)
        assert plan_dict["original_prompt"] == "Do something"
        assert len(plan_dict["sub_prompts"]) == 1

    def test_plan_modification_creates_new_instance(self):
        """Test that modifying Plan creates new instance (Pydantic immutability)."""
        plan1 = Plan(
            original_prompt="Original",
            sub_prompts=[SubPrompt(id=1, content="Task")]
        )
        # Pydantic V2 models are mutable by default, but we can test copying
        plan2 = plan1.model_copy()
        assert plan1 is not plan2
        assert plan1 == plan2