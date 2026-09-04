"""IsolateGPT agent for AgentDojo using LangGraph.

This module provides the main entry point for loading the IsolateGPT agent
in AgentDojo. It implements a Hub-Spoke architecture for tool isolation.
"""

from typing import Any, Literal
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Import LangGraphAgent for wrapping
from agentdojo.agents.base_agent import LangGraphAgent

from .isolate_gpt_creation import create_isolate_gpt_graph


def create_agent(
    model: Literal["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"] = "gpt-4o-mini",
    **kwargs: Any,
) -> LangGraphAgent:
    """Create IsolateGPT agent for AgentDojo.

    This function serves as the main entry point for loading the IsolateGPT agent
    in AgentDojo. It creates a Hub-Spoke architecture agent with tool isolation.

    Args:
        model: Model name to use. Supported models:
            - "gpt-4o-mini": OpenAI GPT-4o mini model
            - "gpt-4o": OpenAI GPT-4o model
            - "claude-3-5-sonnet": Anthropic Claude 3.5 Sonnet
            - "gemini-1.5-pro": Google Gemini 1.5 Pro
        **kwargs: Additional arguments passed by benchmark system:
            - tools (list[Function]): List of tools from suite.tools (required)
            - langchain_tools (list[BaseTool]): LangChain tools (preferred)
            - suite_name (str): Name of the task suite being benchmarked (optional)

    Returns:
        LangGraphAgent wrapping the IsolateGPT compiled graph.

    Raises:
        ValueError: If tools are not provided or if an unsupported model is specified.
    """
    # Extract tools from kwargs (provided by benchmark system)
    langchain_tools = kwargs.get("langchain_tools")
    function_tools = kwargs.get("tools")

    if langchain_tools is not None:
        tools = langchain_tools
    elif function_tools is not None:
        tools = function_tools
    else:
        raise ValueError(
            "Tools must be provided via 'langchain_tools' or 'tools' parameter. "
            "The benchmark system passes both automatically."
        )

    # Convert model string to LLM instance
    llm = _get_model_instance(model)

    # Extract kwargs to pass to create_isolate_gpt_graph
    graph_kwargs = {k: v for k, v in kwargs.items()
                   if k not in ["langchain_tools", "tools", "suite_name", "suite"]}

    # Create the isolate_gpt agent graph with initial tools
    graph = create_isolate_gpt_graph(
        llm=llm,
        tools=tools,
    )

    # Create a graph factory that can rebuild the graph with different tools
    # This is used for environment dependency injection
    def graph_factory(environment_bound_tools: list) -> CompiledStateGraph:
        """Factory function to create a fresh graph with environment-bound tools."""
        return create_isolate_gpt_graph(
            llm=llm,
            tools=environment_bound_tools,
        )

    # Return wrapped agent with both graph and factory
    return LangGraphAgent(
        name="isolate_gpt",
        graph=graph,
        graph_factory=graph_factory,
        model_name=str(model),
    )


def _get_model_instance(
    model_name: Literal["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"]
):
    """Convert model string to LangChain model instance.

    Args:
        model_name: Name of the model to instantiate.

    Returns:
        LangChain model instance configured for the specified provider.

    Raises:
        ValueError: If the model name is not supported.
    """
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model=model_name, temperature=0)
    elif "claude" in model_name.lower():
        if model_name == "claude-3-5-sonnet":
            model_name = "claude-3-5-sonnet-20241022"
        return ChatAnthropic(model=model_name, temperature=0)
    elif "gemini" in model_name.lower():
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
