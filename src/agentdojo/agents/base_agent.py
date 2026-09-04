from typing import Any, Callable, Protocol, runtime_checkable
from langgraph.graph.state import CompiledStateGraph
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement


@runtime_checkable
class BaseAgent(Protocol):
    """Protocol for all agent implementations."""
    name: str


class PipelineAgent(BaseAgent):
    """Agent that uses pipeline execution mode."""

    def __init__(self, name: str, pipeline: BasePipelineElement):
        self.name = name
        self.pipeline = pipeline

    def get_pipeline(self) -> BasePipelineElement:
        """Get the pipeline executor."""
        return self.pipeline


class LangGraphAgent(BaseAgent):
    """Agent that uses LangGraph execution mode."""

    def __init__(
        self,
        name: str,
        graph: CompiledStateGraph,
        graph_factory: Callable[[list], CompiledStateGraph] | None = None,
        model_name: str | None = None,
    ):
        self.name = name
        self.graph = graph
        self.graph_factory = graph_factory
        self.model_name = model_name

    def get_graph(self) -> CompiledStateGraph:
        """Get the compiled graph."""
        return self.graph

    def get_graph_factory(self) -> Callable[[list], CompiledStateGraph] | None:
        """Get the graph factory if available."""
        return self.graph_factory

    def get_model_name(self) -> str | None:
        """Get the model name for attack personalization."""
        return self.model_name


# Type alias for convenience
AgentType = PipelineAgent | LangGraphAgent
