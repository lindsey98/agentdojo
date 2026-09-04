import importlib
import sys
from pathlib import Path
from typing import Any
import yaml

from agentdojo.agents.agent_modes import AgentMode, AgentsConfiguration
from agentdojo.agents.base_agent import AgentType, PipelineAgent, LangGraphAgent
from agentdojo.agents.exceptions import (
    AgentNotFoundError,
    AgentModuleNotFoundError,
    AgentFactoryNotFoundError,
)


class AgentLoader:
    """Loads agents from root-level agents/ directory based on configuration."""

    def __init__(self, config_path: Path | str | None = None, agents_root: Path | str | None = None):
        # Resolve config path
        if config_path is None:
            # From agentdojo/src/agentdojo/agents/ to agentdojo/src/agentdojo/configs/
            module_dir = Path(__file__).parent.parent
            config_path = module_dir / "configs" / "agents.yaml"

        # Load configuration
        with Path(config_path).open() as f:
            config_data = yaml.safe_load(f)
        self.config = AgentsConfiguration.model_validate(config_data)

        # Resolve agents root
        if agents_root is None:
            # Go up from agentdojo/src/agentdojo/agents/ to project root
            module_dir = Path(__file__).parent.parent.parent.parent.parent
            agents_root = module_dir / "agents"

        self.agents_root = Path(agents_root)

        # Add to sys.path for imports
        agents_parent = str(self.agents_root.parent)
        if agents_parent not in sys.path:
            sys.path.insert(0, agents_parent)

    def load_agent(self, agent_name: str, **kwargs: Any) -> AgentType:
        """Load an agent by name.

        Args:
            agent_name: Name of the agent to load (e.g., 'baseline', 'isolategpt')
            **kwargs: Additional arguments passed to create_agent()

        Returns:
            AgentType instance (either PipelineAgent or LangGraphAgent)

        Raises:
            AgentNotFoundError: Agent not in configuration
            AgentModuleNotFoundError: Could not import agent module
            AgentFactoryNotFoundError: Module doesn't export create_agent()
        """
        # Get mode from config
        mode = self.config.get_agent_mode(agent_name)
        if mode is None:
            available = list(self.config.get_all_agents().keys())
            raise AgentNotFoundError(agent_name, available)

        # Determine module name
        module_suffix = "graph" if mode == AgentMode.LANGGRAPH else "pipeline"
        module_name = f"agents.{agent_name}.{module_suffix}"

        # Import module
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise AgentModuleNotFoundError(agent_name, mode.value, module_name, e)

        # Get factory function
        if not hasattr(module, "create_agent"):
            raise AgentFactoryNotFoundError(agent_name, mode.value, module_name)

        # Create agent instance
        agent_instance = module.create_agent(**kwargs)

        # Wrap if needed (support both wrapped and raw return types)
        if mode == AgentMode.LANGGRAPH:
            if not isinstance(agent_instance, LangGraphAgent):
                agent_instance = LangGraphAgent(name=agent_name, graph=agent_instance)
        else:
            if not isinstance(agent_instance, PipelineAgent):
                agent_instance = PipelineAgent(name=agent_name, pipeline=agent_instance)

        return agent_instance


# Global singleton instance
_LOADER: AgentLoader | None = None


def get_loader(config_path: Path | str | None = None, agents_root: Path | str | None = None) -> AgentLoader:
    """Get the global AgentLoader instance (singleton pattern)."""
    global _LOADER
    if _LOADER is None:
        _LOADER = AgentLoader(config_path, agents_root)
    return _LOADER


def load_agent(agent_name: str, **kwargs: Any) -> AgentType:
    """Load an agent by name (convenience function).

    Usage:
        agent = load_agent("baseline", model="gpt-4o")
        graph = agent.get_graph()
    """
    return get_loader().load_agent(agent_name, **kwargs)


def list_agents() -> dict[str, AgentMode]:
    """List all configured agents and their modes."""
    return get_loader().config.get_all_agents()
