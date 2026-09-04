"""LangGraph nodes for Llama Prompt Guard 2 integration.

This module provides the node functions and routing logic for
integrating Llama Prompt Guard 2 into a LangGraph agent.
"""

from typing import Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps

from .huggingface_client import LlamaPromptGuardClient, PromptGuardResult


class PromptGuardAgentState(TypedDict):
    """Extended agent state with prompt guard tracking."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    # Prompt guard additions
    prompt_guard_flagged: list[dict]  # List of flagged tool results


def create_prompt_guard_node(client: LlamaPromptGuardClient):
    """Factory function to create the prompt guard node.

    Args:
        client: LlamaPromptGuardClient instance for classification.

    Returns:
        A node function that checks tool messages for prompt injection.
    """

    def prompt_guard_node(state: PromptGuardAgentState) -> dict:
        """Check tool results for prompt injection attacks.

        This node inspects all recent ToolMessage results and flags
        any that contain malicious content detected by Llama Prompt Guard 2.

        Args:
            state: Current agent state with messages.

        Returns:
            Updated state with flagged messages list.
        """
        messages = state["messages"]
        flagged_messages = []

        # Find all recent ToolMessages (from the last tool execution batch)
        for i, message in enumerate(reversed(messages)):
            if not isinstance(message, ToolMessage):
                # Stop when we hit a non-ToolMessage (likely AIMessage with tool calls)
                break

            # Get the tool result content as string
            content = message.content
            if not isinstance(content, str):
                content = str(content)

            # Skip empty content
            if not content.strip():
                continue

            # Classify the tool result
            try:
                result: PromptGuardResult = client.classify(content)
                print(f"[PromptGuard] Tool: {message.name} | Label: {result.label} | Score: {result.score:.4f} | Malicious: {result.is_malicious}")

                if result.is_malicious:
                    flagged_messages.append({
                        "index": len(messages) - 1 - i,
                        "tool_call_id": message.tool_call_id,
                        "tool_name": message.name,
                        "label": result.label,
                        "score": result.score,
                    })
            except Exception as e:
                # Log error but don't block on classification failures
                print(f"[PromptGuard] Classification error: {e}")
                continue

        return {"prompt_guard_flagged": flagged_messages}

    return prompt_guard_node


def create_prompt_guard_router():
    """Factory for the routing function after prompt guard check.

    Returns:
        A routing function that decides whether to continue to agent or block.
    """

    def route_after_guard(state: PromptGuardAgentState) -> Literal["agent", "blocked"]:
        """Route based on prompt guard detection results.

        Args:
            state: Current agent state with flagged messages.

        Returns:
            "blocked" if malicious content detected, "agent" otherwise.
        """
        flagged = state.get("prompt_guard_flagged", [])

        if flagged:
            return "blocked"

        return "agent"

    return route_after_guard


def create_blocked_node():
    """Factory for the blocked node that returns an error message.

    Returns:
        A node function that returns an error AIMessage.
    """

    def blocked_node(state: PromptGuardAgentState) -> dict:
        """Generate an error message when prompt injection is detected.

        Args:
            state: Current agent state with flagged messages.

        Returns:
            State update with error AIMessage.
        """
        flagged = state.get("prompt_guard_flagged", [])

        # Build error message with details about flagged content
        if flagged:
            flagged_tools = ", ".join(f["tool_name"] for f in flagged)
            error_content = (
                f"I detected potential prompt injection in tool results "
                f"from: {flagged_tools}. Execution has been halted for safety. "
                f"Please review the tool outputs for suspicious content."
            )
        else:
            error_content = (
                "Execution blocked due to detected security issue in tool results."
            )

        return {
            "messages": [AIMessage(content=error_content)]
        }

    return blocked_node
