from rich.console import Console
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Mapping

from shardguard.core.models import Plan
from shardguard.core.planning import PlanningLLM
from shardguard.core.prompts import PLANNING_PROMPT
from shardguard.core.execution import StepExecutor
from shardguard.core.mcp_integration import MCPClient

class CoordinationService:
    """Coordination service for planning."""

    def __init__(self, planner: PlanningLLM):
        self.planner = planner
        self.console = Console()
        # Saving all the args (opaque values) from the prompt into this dictionary 
        self.args: Dict[str, Any] = {}

    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        """
        Normalize SubPrompt into a real dict.
        """
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            return asdict(obj)
        # Added this as most of the tool objects are being referenced as Sharguard Model Schema
        if hasattr(obj, "model_dump") and callable(obj.model_dump): 
            return obj.model_dump()  # Pydantic v2
        if hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()        # Pydantic v1
        if isinstance(obj, Mapping):
            return dict(obj)
        if hasattr(obj, "__dict__"):
            return dict(vars(obj))
        raise TypeError(f"Unsupported step type: {type(obj)!r}. Provide a dict-like object.")

    async def handle_prompt(self, user_input: str) -> Plan:
        formatted_prompt = self._format_prompt(user_input)
        plan_json = await self.planner.generate_plan(formatted_prompt)
        return Plan.model_validate_json(plan_json)

    def _format_prompt(self, user_input: str) -> str:
        """Format the user input using the planning prompt template."""
        return PLANNING_PROMPT.format(user_prompt=user_input)
    
    def extract_arguments(self, task):
        """
        Extracting arguments from the prompt for both cases:
            1. Getting both key-value pairs for the system as a whole
            2. Getting only the key for the parameter to be obfuscated
        """
        opaque = task.get("opaque_values") or {}
        if not isinstance(opaque, Mapping):
            return []
        for k, v in opaque.items():
            self.args[k] = v
        return list(opaque.keys())

    async def execute_tools(self, LLMStepResponse, argument_dicts):
        """
        Function to make calls to specific tools specified by the Planning LLM
        Args:
            LLMStepResponse: Processed prompts to breakdown specific tasks so that no other MCP knows about each other
            argument_dicts: The dictionary of arguments whose value would be required by the tool to execute the task
        """
        mcp = MCPClient()
        # Resolved values will contain the exact value of the argument instead of the Opaque ones
        resolved_values = {}
        if argument_dicts:
            for arg_dict in argument_dicts:
                if arg_dict in self.args:
                    resolved_values[arg_dict] = self.args[arg_dict]
                    break

        for calls in LLMStepResponse.tool_calls:
            await MCPClient.call_tool(
                mcp,
                calls.server,
                calls.tool,
                resolved_values
            )
            print(f"Executed {calls.server}: {calls.tool} with {argument_dicts}")
        return
    
    async def handle_subtasks(self, tasks):
        """Sends the subtasks to ExecutionLLM"""
        for task in tasks:
            task = self._to_dict(task)
            argument_dicts = self.extract_arguments(task)
            # Sends the task to process for execution
            LLMStepResponse = await StepExecutor.run_step(self, task)
            await self.execute_tools(LLMStepResponse, argument_dicts)
        return

