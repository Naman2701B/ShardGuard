"""Execution LLM with multiple provider support and intent validation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os

from jsonschema import validate as jsonschema_validate
from abc import ABC, abstractmethod

from shardguard.core.schemas import TOOL_INTENTS_SCHEMA
from shardguard.core.llm_providers import LLMProviderFactory

logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """Represents a tool call proposal for safe execution."""
    server: str
    tool: str
    args: Optional[Dict[str, Any]] = None

@dataclass
class LLMStepResponse:
    """Response for a single execution step containing proposed tool calls."""
    tool_calls: List[ToolCall]

class ExecutionLLM(ABC):
    """Abstract base class for execution-oriented LLMs."""

    @abstractmethod
    async def propose_tool_intents(
        self,
        *,
        step_content: str,
        nonsecret_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Propose validated tool intents based on the step content."""

EXEC_SYSTEM_PROMPT = (
    "You are the Execution LLM inside ShardGuard.\n"
    "Output ONLY a JSON array of tool intents. No text, no code fences.\n"
    'Each item: {"server": "...", "tool": "...", "args": { ...optional... }}\n'
    "Do not include secrets, credentials, or opaque tokens."
)

def _build_exec_prompt(task: str, ctx: Optional[Dict[str, Any]]) -> str:
    """Builds a consistent JSON-based prompt for all execution providers."""
    print(ctx)
    ctx_json = json.dumps(ctx or {}, ensure_ascii=False)
    return (
        f"{EXEC_SYSTEM_PROMPT}\n\n"
        f"Task:\n{task or ''}\n\n"
        f"Context (JSON):\n{ctx_json}\n\n"
        "Return ONLY a JSON array (no prose)."
    )

def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Parse and validate a JSON array from LLM output.
    Falls back to regex extraction if model includes extra text.
    """
    if not isinstance(text, str):
        raise ValueError("Execution LLM returned non-string content.")

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            jsonschema_validate(obj, TOOL_INTENTS_SCHEMA)
            return obj
    except Exception:
        pass

    match = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        intents = json.loads(candidate)
        jsonschema_validate(intents, TOOL_INTENTS_SCHEMA)
        return intents

    # If no valid output, return an empty list (valid per schema)
    jsonschema_validate([], TOOL_INTENTS_SCHEMA)
    return []

class GenericExecutionLLM(ExecutionLLM):
    """
    Unified Execution LLM supporting multiple providers (Gemini, Ollama, etc.).
    Handles prompt construction, response parsing, and validation.
    """

    def __init__(
        self,
        provider_type: str = "ollama",
        model: str = os.getenv("OLLAMA_MODEL"),
        base_url: str = os.getenv("BASE_URL"),
        api_key: Optional[str] = None,
    ):
        self.provider_type = provider_type
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

        provider_kwargs = {}
        if provider_type.lower() == "ollama":
            provider_kwargs["base_url"] = base_url
        elif provider_type.lower() == "gemini":
            provider_kwargs["api_key"] = api_key

        self.llm_provider = LLMProviderFactory.create_provider(
            provider_type=provider_type,
            model=model,
            **provider_kwargs,
        )

    async def propose_tool_intents(
        self,
        *,
        step_content: str,
        nonsecret_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate tool intents using the configured LLM provider."""
        prompt = _build_exec_prompt(step_content, nonsecret_context)

        try:
            raw_response = await self.llm_provider.generate_response(prompt)
            text = raw_response if isinstance(raw_response, str) else json.dumps(raw_response)
            return _extract_json_array(text)

        except Exception as e:
            logger.error(f"[{self.provider_type}] Execution error: {e}")
            return []

    def close(self):
        """Close provider connections (if applicable)."""
        self.llm_provider.close()

class StepExecutor:
    """Executes steps using a configured Execution LLM."""

    def __init__(self, exec_llm: ExecutionLLM):
        self.exec_llm = exec_llm

    async def run_step(self, step: Dict[str, Any]) -> LLMStepResponse:
        """Run a single step through the LLM to get validated tool calls."""
        calls: List[ToolCall] = []

        nonsecret_ctx = step.get("nonsecret_context") or {}
        intents = await self.exec_llm.propose_tool_intents(
            step_content=step.get("content", ""),
            nonsecret_context=nonsecret_ctx,
        )

        logger.debug(f"Proposed tool intents: {json.dumps(intents, indent=2)}")

        for it in intents:
            calls.append(
                ToolCall(server=it["server"], tool=it["tool"], args=it.get("args"))
            )

        return LLMStepResponse(tool_calls=calls)

def make_execution_llm(
    provider_type: str = "ollama",
    model: str = os.getenv("OLLAMA_MODEL"),
    base_url: str = os.getenv("BASE_URL"),
    api_key: Optional[str] = None,
) -> GenericExecutionLLM:
    """
    Factory function to create a GenericExecutionLLM instance
    with the correct provider initialized.
    """
    return GenericExecutionLLM(
        provider_type=provider_type,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )