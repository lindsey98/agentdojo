class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class AgentNotFoundError(AgentError):
    """Agent not found in configuration."""

    def __init__(self, agent_name: str, available_agents: list[str] | None = None):
        self.agent_name = agent_name
        self.available_agents = available_agents
        message = f"Agent '{agent_name}' not found."
        if available_agents:
            message += f" Available agents: {', '.join(available_agents)}"
        super().__init__(message)


class AgentModuleNotFoundError(AgentError):
    """Agent module could not be imported."""

    def __init__(self, agent_name: str, mode: str, module_path: str, original_error: Exception):
        message = f"Could not import {mode} module for agent '{agent_name}' from '{module_path}': {original_error}"
        super().__init__(message)


class AgentFactoryNotFoundError(AgentError):
    """Agent module does not export create_agent() function."""

    def __init__(self, agent_name: str, mode: str, module_path: str):
        message = f"Module '{module_path}' for agent '{agent_name}' ({mode} mode) does not export 'create_agent' function"
        super().__init__(message)
