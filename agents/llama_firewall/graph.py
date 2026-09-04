"""LlamaFirewall AlignmentCheck protected ReAct agent for AgentDojo using LangGraph.

This module provides a baseline agent protected by LlamaFirewall's AlignmentCheck
scanner for detecting misaligned agent behavior as an output guardrail.
"""

from typing import Any, Literal, Optional, Sequence, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from langgraph.utils.runnable import RunnableCallable
from typing_extensions import Annotated, TypedDict

# Import from baseline for reuse
from agentdojo.agents.base_agent import LangGraphAgent

# Import baseline utilities
from agents.baseline.utils import bind_tools
from agents.baseline.react_agent_creation import (
    MessagesModifier,
    StateModifier,
    _get_model_preprocessing_runnable,
    _should_bind_tools,
)

# Import alignment check components
from .alignment_check_node import (
    AlignmentCheckAgentState,
    create_alignment_check_node,
    create_alignment_router,
    create_blocked_node,
)
from .llamafirewall_client import AlignmentCheckClient


class AgentState(TypedDict):
    """The state of the agent with alignment check support."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    alignment_check_result: Optional[dict]


def create_react_agent_with_alignment_check(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool], ToolNode],
    alignment_check_client: AlignmentCheckClient,
    *,
    state_schema: Optional[type] = None,
    messages_modifier: Optional[MessagesModifier] = None,
    state_modifier: Optional[StateModifier] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    parallel_tool_calls: bool = True,
) -> CompiledStateGraph:
    """Create a ReAct agent with LlamaFirewall AlignmentCheck protection.

    This extends the baseline ReAct agent to add an alignment check
    node AFTER the agent generates a response (output guardrail).

    Graph flow:
        agent -> alignment_check -> route -> tools/__end__/blocked

    Args:
        model: The LLM to use for the agent.
        tools: The tools available to the agent.
        alignment_check_client: Client for LlamaFirewall AlignmentCheck.
        state_schema: Optional custom state schema.
        messages_modifier: Optional modifier for messages.
        state_modifier: Optional modifier for state.
        checkpointer: Optional checkpointer for persistence.
        store: Optional store for state.
        interrupt_before: Optional list of nodes to interrupt before.
        interrupt_after: Optional list of nodes to interrupt after.
        debug: Whether to enable debug mode.
        parallel_tool_calls: Whether to allow parallel tool calls.

    Returns:
        Compiled LangGraph state graph with alignment check protection.
    """
    # Disable parallel tool calls for Gemini
    if isinstance(model, ChatGoogleGenerativeAI):
        parallel_tool_calls = False

    if state_schema is not None:
        if missing_keys := {"messages", "is_last_step"} - set(
            state_schema.__annotations__
        ):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    # Remove _u tools which are only for PFI agent
    tools_list = list(tools) if not isinstance(tools, ToolNode) else list(tools.tools_by_name.values())
    tools_list = [t for t in tools_list if not t.name.endswith("_u")]

    tool_node = ToolNode(tools_list, handle_tool_errors=True)
    tool_classes = list(tool_node.tools_by_name.values())

    if _should_bind_tools(model, tool_classes):
        model = bind_tools(model, tool_classes, enable_parallel_tool_calls=parallel_tool_calls)

    def tool_node_wrapper(state: AgentState) -> dict:
        tool_node_inner = ToolNode(tools_list, handle_tool_errors=True)

        # If parallel tool call is disabled, remove all tool calls except the first one (For Gemini)
        if not parallel_tool_calls:
            for message in reversed(state["messages"]):
                if message.type == "ai":
                    ai_tool_call = message
                    break
            ai_tool_call.tool_calls = ai_tool_call.tool_calls[:1]

        return tool_node_inner.invoke(state)

    # Get model preprocessing runnable
    preprocessor = _get_model_preprocessing_runnable(
        state_modifier, messages_modifier, store
    )
    model_runnable = preprocessor | model

    # Track tools that should return directly
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    # Define the function that calls the model
    def call_model(state: AgentState, config: RunnableConfig) -> dict:
        response = model_runnable.invoke(state, config)
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        if (
            (
                "remaining_steps" not in state
                and state["is_last_step"]
                and has_tool_calls
            )
            or (
                "remaining_steps" in state
                and state["remaining_steps"] < 1
                and all_tools_return_direct
            )
            or (
                "remaining_steps" in state
                and state["remaining_steps"] < 2
                and has_tool_calls
            )
        ):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        return {"messages": [response]}

    async def acall_model(state: AgentState, config: RunnableConfig) -> dict:
        response = await model_runnable.ainvoke(state, config)
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        if (
            (
                "remaining_steps" not in state
                and state["is_last_step"]
                and has_tool_calls
            )
            or (
                "remaining_steps" in state
                and state["remaining_steps"] < 1
                and all_tools_return_direct
            )
            or (
                "remaining_steps" in state
                and state["remaining_steps"] < 2
                and has_tool_calls
            )
        ):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        return {"messages": [response]}

    # Create alignment check nodes
    alignment_check_node = create_alignment_check_node(alignment_check_client)
    route_after_check = create_alignment_router()
    blocked_node = create_blocked_node()

    # Define a new graph
    workflow = StateGraph(state_schema or AgentState)

    # Define the nodes
    workflow.add_node("agent", RunnableCallable(call_model, acall_model))
    workflow.add_node("alignment_check", alignment_check_node)
    workflow.add_node("tools", tool_node_wrapper)
    workflow.add_node("blocked", blocked_node)

    # Set the entrypoint as `agent`
    workflow.set_entry_point("agent")

    # Route from agent: if has tool_calls -> tools, else -> alignment_check
    def route_after_agent(state: AgentState) -> Literal["tools", "alignment_check"]:
        """Route based on whether agent wants to use tools.

        Only run alignment check when agent is done with tools (final response).
        This prevents O(n²) API calls during multi-turn tool usage.
        """
        messages = state["messages"]
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
        return "alignment_check"

    workflow.add_conditional_edges("agent", route_after_agent)

    # alignment_check -> __end__ or blocked (no more "tools" option here)
    workflow.add_conditional_edges("alignment_check", route_after_check)

    # Tools -> agent (loop back to agent after tool execution)
    workflow.add_edge("tools", "agent")

    # blocked -> __end__
    workflow.add_edge("blocked", "__end__")

    # Compile the graph
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )


def _get_model_instance(model_name: str):
    """Convert model string to LangChain model instance."""
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


def create_agent(
    model: str = "gpt-4o-mini",
    **kwargs: Any,
) -> LangGraphAgent:
    """Create LlamaFirewall AlignmentCheck protected agent for AgentDojo.

    This function serves as the main entry point for loading the agent.
    It creates a baseline ReAct agent with LlamaFirewall AlignmentCheck
    protection for detecting misaligned agent behavior (output guardrail).

    Args:
        model: Model name to use.
        **kwargs: Additional arguments:
            - langchain_tools / tools: Tools for the agent (required)

    Returns:
        LangGraphAgent with alignment check protection.
    """
    # Extract tools from kwargs
    langchain_tools = kwargs.get("langchain_tools")
    function_tools = kwargs.get("tools")

    if langchain_tools is not None:
        tools = langchain_tools
    elif function_tools is not None:
        tools = function_tools
    else:
        raise ValueError(
            "Tools must be provided via 'langchain_tools' or 'tools' parameter."
        )

    # Create alignment check client (requires TOGETHER_API_KEY env var)
    alignment_check_client = AlignmentCheckClient()

    # Get LLM
    llm = _get_model_instance(model)

    # Extract kwargs to pass to create_react_agent_with_alignment_check
    react_kwargs = {k: v for k, v in kwargs.items()
                   if k not in ["langchain_tools", "tools", "suite_name", "suite"]}

    # Create the graph
    graph = create_react_agent_with_alignment_check(
        model=llm,
        tools=tools,
        alignment_check_client=alignment_check_client,
        **react_kwargs
    )

    # Create factory for environment binding
    def graph_factory(environment_bound_tools: list) -> CompiledStateGraph:
        return create_react_agent_with_alignment_check(
            model=llm,
            tools=environment_bound_tools,
            alignment_check_client=alignment_check_client,
            **react_kwargs
        )

    return LangGraphAgent(
        name="llama_firewall",
        graph=graph,
        graph_factory=graph_factory,
        model_name=str(model),
    )
