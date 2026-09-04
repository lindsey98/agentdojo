"""LangGraph nodes for LlamaFirewall AlignmentCheck integration.

This module provides the node functions and routing logic for
integrating LlamaFirewall AlignmentCheck into a LangGraph agent.
"""

from typing import Literal, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from typing_extensions import Annotated, TypedDict

from .llamafirewall_client import AlignmentCheckClient, AlignmentCheckResult


class AlignmentCheckAgentState(TypedDict):
    """Extended agent state with alignment check tracking."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    # Alignment check additions
    alignment_check_result: Optional[dict]  # Result from alignment check


def create_alignment_check_node(client: AlignmentCheckClient):
    """Factory function to create the alignment check node.

    Args:
        client: AlignmentCheckClient instance for checking alignment.

    Returns:
        A node function that checks the conversation for alignment issues.
    """

    def alignment_check_node(state: AlignmentCheckAgentState) -> dict:
        """Check conversation trace for alignment issues.

        This node inspects the full conversation history and checks
        for potential misalignment using LlamaFirewall's AlignmentCheck.

        Args:
            state: Current agent state with messages.

        Returns:
            Updated state with alignment check result.
        """
        messages = state["messages"]

        try:
            result: AlignmentCheckResult = client.check_alignment(list(messages))
            print(
                f"[AlignmentCheck] Decision: {result.decision} | "
                f"Score: {result.score:.4f} | Misaligned: {result.is_misaligned}"
            )
            if result.reason:
                print(f"[AlignmentCheck] Reason: {result.reason[:200]}...")

            return {
                "alignment_check_result": {
                    "is_misaligned": result.is_misaligned,
                    "decision": result.decision,
                    "reason": result.reason,
                    "score": result.score,
                }
            }
        except Exception as e:
            # Log error but don't block on check failures
            print(f"[AlignmentCheck] Error: {e}")
            return {
                "alignment_check_result": {
                    "is_misaligned": False,
                    "decision": "ALLOW",
                    "reason": f"Check failed: {e}",
                    "score": 0.0,
                }
            }

    return alignment_check_node


def create_alignment_router():
    """Factory for the routing function after alignment check.

    Returns:
        A routing function that decides whether to end or block.
    """

    def route_after_alignment_check(
        state: AlignmentCheckAgentState
    ) -> Literal["__end__", "blocked"]:
        """Route based on alignment check results.

        Note: This is only called when agent has no more tool calls (final response).
        The tool routing is handled separately in the graph.

        Args:
            state: Current agent state with alignment check result.

        Returns:
            "blocked" if misaligned, otherwise "__end__".
        """
        result = state.get("alignment_check_result", {})

        # Check if misaligned
        if result.get("is_misaligned", False):
            return "blocked"

        return "__end__"

    return route_after_alignment_check


def create_blocked_node():
    """Factory for the blocked node that returns an error message.

    Returns:
        A node function that returns an error AIMessage.
    """

    def blocked_node(state: AlignmentCheckAgentState) -> dict:
        """Generate an error message when misalignment is detected.

        Args:
            state: Current agent state with alignment check result.

        Returns:
            State update with error AIMessage.
        """
        result = state.get("alignment_check_result", {})

        reason = result.get("reason", "Unknown reason")
        decision = result.get("decision", "BLOCK")
        score = result.get("score", 1.0)

        error_content = (
            f"I detected potential misalignment in my behavior. "
            f"Decision: {decision} (score: {score:.2f}). "
            f"Execution has been halted for safety. "
            f"Reason: {reason[:500] if reason else 'No reason provided'}"
        )

        return {
            "messages": [AIMessage(content=error_content)]
        }

    return blocked_node
