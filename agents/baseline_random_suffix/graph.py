"""Baseline ReAct agent with Random Suffix Defense for AgentDojo using LangGraph.

This module provides the main entry point for loading the baseline agent with
random suffix defense in AgentDojo. It wraps the core react_agent_creation logic
with an AgentDojo-compliant interface.

The random suffix defense adds random suffixes to all JSON field names in tool
results (e.g., status -> status_x7k9m2) and injects system instructions to guide
the model to only trust keys with the random suffix.
"""

from typing import Any, Literal
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Import LangGraphAgent for wrapping
from agentdojo.agents.base_agent import LangGraphAgent

from .react_agent_creation import create_react_agent


def create_agent(
    model: Literal["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"] = "gpt-4o-mini",
    **kwargs: Any,
) -> LangGraphAgent:
    """Create baseline ReAct agent for AgentDojo.

    This function serves as the main entry point for loading the baseline agent
    in AgentDojo. It converts the model string to an LLM instance and delegates
    to create_react_agent for the actual graph construction.

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
            Other custom arguments can be added for agent-specific configuration.

    Returns:
        Compiled LangGraph state graph ready for execution.

    Raises:
        ValueError: If tools are not provided or if an unsupported model is specified.

    Example:
        >>> from agentdojo.agents import load_agent
        >>> agent = load_agent("baseline_random_suffix", model="gpt-4o-mini", tools=[...])
        >>> graph = agent.get_graph()
        >>> result = graph.invoke({"messages": [HumanMessage("Hello")]})
    """
    # Extract tools from kwargs (provided by benchmark system)
    # Prefer langchain_tools if available (for LangGraph compatibility)
    langchain_tools = kwargs.get("langchain_tools")
    function_tools = kwargs.get("tools")

    if langchain_tools is not None:
        tools = langchain_tools  # Use LangChain tools (preferred for LangGraph)
    elif function_tools is not None:
        tools = function_tools  # Fallback to Function objects
    else:
        raise ValueError(
            "Tools must be provided via 'langchain_tools' or 'tools' parameter. "
            "The benchmark system passes both automatically."
        )

    # Extract suite_name if provided (optional, for suite-specific behavior)
    suite_name = kwargs.get("suite_name")

    # Convert model string to LLM instance
    llm = _get_model_instance(model)

    # Extract kwargs to pass to create_react_agent
    # Remove tool parameters from kwargs since we pass tools as a positional argument
    react_kwargs = {k: v for k, v in kwargs.items()
                   if k not in ["langchain_tools", "tools", "suite_name", "suite"]}

    # Create the react agent graph with initial tools
    graph = create_react_agent(
        model=llm,
        tools=tools,
        **react_kwargs
    )

    # Create a graph factory that can rebuild the graph with different tools
    # This is used for environment dependency injection
    def graph_factory(environment_bound_tools: list) -> CompiledStateGraph:
        """Factory function to create a fresh graph with environment-bound tools."""
        return create_react_agent(
            model=llm,
            tools=environment_bound_tools,
            **react_kwargs
        )

    # Return wrapped agent with both graph and factory
    return LangGraphAgent(
        name="baseline_random_suffix",
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

    Note:
        Currently uses default API key from environment variables.
    """
    # Handle OpenAI models (including version-specific model names)
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model=model_name, temperature=0)
    # Handle Anthropic Claude models
    elif "claude" in model_name.lower():
        # Use the specific model name provided, or default to versioned model
        if model_name == "claude-3-5-sonnet":
            model_name = "claude-3-5-sonnet-20241022"
        return ChatAnthropic(model=model_name, temperature=0)
    # Handle Google Gemini models
    elif "gemini" in model_name.lower():
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
