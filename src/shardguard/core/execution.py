# shardguard/core/execution.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Mapping

@dataclass
class ToolCall:
    """
    This contains only the server and tool required which would be returned,
    args would not be part of this as the execution LLM is assumed to not be
    trusted
    """
    server: str
    tool: str

@dataclass
class LLMStepResponse:
    """
    This contains the list of ToolCall, as multiple calls might be needed to make a specific sub prompt run.
    """
    tool_calls: List[ToolCall] = field(default_factory=list)

class StepExecutor:
    """
    This gets the 
    - Returns normalized tool calls.
    """

    def __init__(self) -> None:
        pass  # no _deny needed

    async def run_step(self, step: Any) -> LLMStepResponse:

        # Accept both 'suggested_tools' and legacy 'calls'
        calls_spec = step.get("suggested_tools")

        # Normalize to a list; accept single object or "server.tool" string
        if calls_spec is None:
            calls_spec = []
        elif isinstance(calls_spec, Mapping) or hasattr(calls_spec, "__dict__"):
            calls_spec = [calls_spec]
        elif isinstance(calls_spec, str):
            calls_spec = [calls_spec]
        elif not isinstance(calls_spec, list):
            raise ValueError("tool list must be a 'server.tool' string")

        # Make a list of toolcall for a case where multiple tool calls required for a single sub prompt to be processed.
        calls: List[ToolCall] = []

        for i, spec in enumerate(calls_spec):
            server: str
            tool: str

            if isinstance(spec, str):
                st = spec.strip()
                server, tool = st.split(".", 1)
            else:
                if not isinstance(spec, Mapping):
                    if hasattr(spec, "__dict__"):
                        spec = vars(spec)
                    else:
                        raise ValueError(f"suggested_tools[{i}] must be a 'server.tool' string")
            
            calls.append(ToolCall(server=server, tool=tool))

        return LLMStepResponse(tool_calls=calls)