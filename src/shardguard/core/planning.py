"""Planning LLM with MCP integration and multiple provider support."""

import json
import logging
import re
from typing import Protocol

from .llm_providers import LLMProviderFactory
from .mcp_integration import MCPClient

logger = logging.getLogger(__name__)


class PlanningLLMProtocol(Protocol):
    """Protocol for planning LLM implementations."""

    async def generate_plan(self, prompt: str) -> str: ...


class PlanningLLM:
    """Planning LLM with MCP integration and multiple provider support."""

    def __init__(
        self,
        provider_type: str = "ollama",
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
    ):
        """Initialize with MCP client integration and configurable LLM provider."""
        self.provider_type = provider_type
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.mcp_client = MCPClient()

        # Create the appropriate LLM provider
        provider_kwargs = {}
        if provider_type.lower() == "ollama":
            provider_kwargs["base_url"] = base_url
        elif provider_type.lower() == "gemini":
            provider_kwargs["api_key"] = api_key

        self.llm_provider = LLMProviderFactory.create_provider(
            provider_type=provider_type, model=model, **provider_kwargs
        )

    async def generate_plan(self, prompt: str) -> str:
        """Generate a plan using the configured LLM provider."""
        tools_description = await self.mcp_client.get_tools_description()

        # Create enhanced prompt with tools
        enhanced_prompt = (
            f"{prompt}\n\n{tools_description}"
            if tools_description != "No MCP tools available."
            else prompt
        )

        logger.debug("Full prompt sent to model:\n%s", enhanced_prompt)

        try:
            raw_response = await self.llm_provider.generate_response(enhanced_prompt)
            return self._extract_json_from_response(raw_response)
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return self._create_fallback_response(prompt, str(e))

    async def get_available_tools_description(self) -> str:
        """Get formatted description of all available MCP tools."""
        return await self.mcp_client.get_tools_description()

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response that might contain extra text."""
        # Try to find JSON block enclosed in curly braces
        matches = re.findall(r"\{.*\}", response, re.DOTALL)

        if matches:
            # Return the longest JSON-like match
            json_candidate = max(matches, key=len)
            # Validate that it's actually valid JSON
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass

        # If no valid JSON found, return the original response
        return response

    def _create_fallback_response(self, prompt: str, error: str) -> str:
        """Create a fallback response when plan generation fails."""
        return json.dumps(
            {
                "original_prompt": prompt,
                "sub_prompts": [
                    {
                        "id": 1,
                        "content": f"Error occurred: {error}",
                        "opaque_values": {},
                        "suggested_tools": [],
                    }
                ],
            }
        )

    def close(self):
        """Close any open connections."""
        self.llm_provider.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
