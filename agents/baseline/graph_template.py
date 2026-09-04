"""Baseline agent implementation using LangGraph.

This is a template showing the expected interface for LangGraph agents.
Each agent module must export a create_agent() function that returns
a CompiledStateGraph.
"""

from typing import Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


# Define agent state
class AgentState(TypedDict):
    """State for the baseline agent."""
    messages: list[BaseMessage]


def create_agent(
    model: Literal["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"] = "gpt-4o-mini",
    **kwargs: Any,
) -> CompiledStateGraph:
    """Create baseline LangGraph agent.

    This is a template. Replace with actual implementation.

    Args:
        model: Model name to use.
        **kwargs: Additional arguments for agent configuration.

    Returns:
        Compiled LangGraph state graph ready for execution.

    Example:
        >>> graph = create_agent(model="gpt-4o")
        >>> result = graph.invoke({"messages": [HumanMessage("Hello")]})
    """
    # TODO: Implement actual baseline agent logic
    # This is a placeholder showing the expected return type

    workflow = StateGraph(AgentState)

    # Add nodes and edges here
    # workflow.add_node("agent", agent_node)
    # workflow.set_entry_point("agent")
    # ...

    return workflow.compile()
