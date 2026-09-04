from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class AgentMode(str, Enum):
    """Execution mode for agents."""
    PIPELINE = "pipeline"
    LANGGRAPH = "langgraph"


class AgentsConfiguration(BaseModel):
    """Configuration for all agents and their modes."""
    langgraph_agents: list[str] = Field(default_factory=list)
    pipeline_agents: list[str] = Field(default_factory=list)
    default_mode: Literal["pipeline", "langgraph"] = Field(default="pipeline")

    def get_agent_mode(self, agent_name: str) -> AgentMode | None:
        """Determine the mode for a given agent."""
        if agent_name in self.langgraph_agents:
            return AgentMode.LANGGRAPH
        elif agent_name in self.pipeline_agents:
            return AgentMode.PIPELINE
        return None

    def get_all_agents(self) -> dict[str, AgentMode]:
        """Get all configured agents and their modes."""
        return {
            **{name: AgentMode.LANGGRAPH for name in self.langgraph_agents},
            **{name: AgentMode.PIPELINE for name in self.pipeline_agents},
        }
