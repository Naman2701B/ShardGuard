from pydantic import BaseModel, Field
from typing import Any, List


class SubPrompt(BaseModel):
    id: int
    content: str
    opaque_values: dict[str, str] = Field(default_factory=dict)
    suggested_tools: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    original_prompt: str
    sub_prompts: list[SubPrompt]

# Step Model for breaking the subprompts into executable steps
class Step(BaseModel):
    id: str
    description: str
    tools: List[Any] = []  # objects with .server and .name
