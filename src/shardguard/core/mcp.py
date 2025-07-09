
class MCPServerManager:
    def __init__(self):
        self.servers = {}

    def add_server(self, server_id: str, url: str):
        self.servers[server_id] = url

    def remove_server(self, server_id: str):
        if server_id in self.servers:
            del self.servers[server_id]

    def get_server(self, server_id: str):
        return self.servers.get(server_id)

