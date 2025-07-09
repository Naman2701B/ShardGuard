from shardguard.core.mcp import MCPServerManager
from shardguard.core.planning import PlanningLLM

class ModelWithMCP:
    def __init__(self, mcp_server_manager: MCPServerManager, model: PlanningLLM):
        self.mcp_server_manager = mcp_server_manager
        self.model = model

    def fetch_and_integrate_context(self, server_id: str, user_input: str):
        # Fetch context from the MCP server
        context = self.mcp_server_manager.get_server(server_id)
        
        if context:
            # Incorporate the context into the prompt
            combined_input = f"{user_input} {context}"
            return self.model.generate_plan(combined_input)
        else:
            raise ValueError(f"MCP server with ID {server_id} not found")
