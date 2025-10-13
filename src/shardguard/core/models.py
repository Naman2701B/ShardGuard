# This file has architecture of how the Plan and SubPrompt would look like.

from pydantic import BaseModel, Field


class SubPrompt(BaseModel):
    id: int
    content: str
    opaque_values: dict[str, str] = Field(default_factory=dict)
    suggested_tools: list[str] = Field(default_factory=list)

class Plan(BaseModel):
    original_prompt: str
    sub_prompts: list[SubPrompt]