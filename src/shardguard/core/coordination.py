from rich.console import Console

from shardguard.core.models import Plan
from shardguard.core.planning import PlanningLLM
from shardguard.core.prompts import PLANNING_PROMPT


class CoordinationService:
    """Coordination service for planning."""

    def __init__(self, planner: PlanningLLM):
        self.planner = planner
        self.console = Console()

    async def handle_prompt(self, user_input: str) -> Plan:
        formatted_prompt = self._format_prompt(user_input)
        plan_json = await self.planner.generate_plan(formatted_prompt)
        return Plan.model_validate_json(plan_json)

    def _format_prompt(self, user_input: str) -> str:
        """Format the user input using the planning prompt template."""
        return PLANNING_PROMPT.format(user_prompt=user_input)
