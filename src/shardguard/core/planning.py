"""Planning LLM with MCP integration and multiple provider support."""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Protocol

from shardguard.core.mcp_client import PROJECT_ROOT
from shardguard.mcp_servers import registry

from .llm_providers import LLMProviderFactory

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
        root = Path(PROJECT_ROOT).resolve()
        if root.name == "core":
            root = root.parent.parent
        elif root.name == "shardguard":
            root = root.parent
        elif root.name == "src":
            root = root.parent

        self.registry_path = str(
            root / "src" / "shardguard" / "mcp_servers" / "mcp_registry.json"
        )

        """Initialize with MCP client integration and configurable LLM provider."""
        self.provider_type = provider_type
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

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
        tools_description = await self.get_available_tools_description()

        # Create enhanced prompt with tools
        enhanced_prompt = (
            f"### Available MCP Servers & Tools ###\n{tools_description}\n\n"
            f"### User Request ###\n{prompt}"
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
        registry._CLIENTS.clear()
        tools_map = await asyncio.to_thread(
            registry.fetch_all_tools, self.registry_path
        )

        try:
            reg_data = await asyncio.to_thread(
                registry.load_registry, self.registry_path
            )
            mcps: dict[str, dict[str, Any]] = reg_data.get("mcps", {})

            if not mcps:
                logger.warning(f"No mcps found in registry at {self.registry_path}")
                return "No MCP Servers registered."

            tools_map: dict[str, list[dict[str, Any]]] = await asyncio.to_thread(
                registry.fetch_all_tools, self.registry_path
            )

            lines: list[str] = []
            for server_name in mcps.keys():
                lines.append(f"MCP_SERVER: {server_name}")

                server_tools = tools_map.get(server_name) or []
                if not server_tools:
                    lines.append("  (Status: Offline or No Tools Found)\n")
                    continue

                for tool in server_tools:
                    t_name = tool.get("name") or ""
                    t_desc = tool.get("description") or "No description provided."
                    lines.append(f"  - TOOL_KEY: {t_name}")
                    lines.append(f"    TOOL_DESCRIPTION: {t_desc}")

                    schema = tool.get("inputSchema")
                    if schema:
                        lines.append(
                            f"    TOOL_SCHEMA: {json.dumps(schema, sort_keys=True)}"
                        )

                lines.append("")  # spacer

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Failed to fetch tools: {e}")
            return f"Error loading tools from registry: {e}"

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
