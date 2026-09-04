"""LlamaFirewall SDK client for AlignmentCheck scanner.

This module provides a client wrapper for the LlamaFirewall SDK
to check agent alignment using the AlignmentCheck scanner.
"""

from typing import NamedTuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from llamafirewall import (
    AssistantMessage,
    LlamaFirewall,
    Role,
    ScannerType,
    UserMessage,
)


class AlignmentCheckResult(NamedTuple):
    """Result from AlignmentCheck scanner."""
    is_misaligned: bool
    decision: str  # "ALLOW", "BLOCK", or "HUMAN_IN_THE_LOOP_REQUIRED"
    reason: str
    score: float


class AlignmentCheckClient:
    """Client for LlamaFirewall AlignmentCheck scanner.

    This client uses the LlamaFirewall SDK to analyze conversation
    traces for agent alignment issues.

    Note: Requires TOGETHER_API_KEY environment variable.
    """

    def __init__(self):
        """Initialize the AlignmentCheck client.

        The client requires TOGETHER_API_KEY to be set in the environment.
        """
        self.firewall = LlamaFirewall({
            Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
        })

    def convert_langchain_messages(self, messages: list[BaseMessage]) -> list:
        """Convert LangChain messages to LlamaFirewall trace format.

        Args:
            messages: List of LangChain BaseMessage objects.

        Returns:
            List of LlamaFirewall message objects (UserMessage, AssistantMessage).
        """
        trace = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                trace.append(UserMessage(content=str(msg.content)))
            elif isinstance(msg, AIMessage):
                # Include tool calls in content if present
                content = str(msg.content) if msg.content else ""
                if msg.tool_calls:
                    tool_calls_str = ", ".join(
                        f"{tc['name']}({tc.get('args', {})})" for tc in msg.tool_calls
                    )
                    if content:
                        content = f"{content}\n[Tool calls: {tool_calls_str}]"
                    else:
                        content = f"[Tool calls: {tool_calls_str}]"
                if content:
                    trace.append(AssistantMessage(content=content))
            elif isinstance(msg, ToolMessage):
                # Include tool results as part of the trace
                tool_content = f"[Tool result from {msg.name}]: {str(msg.content)}"
                trace.append(AssistantMessage(content=tool_content))
            elif isinstance(msg, SystemMessage):
                # System messages are typically the initial instruction
                trace.append(UserMessage(content=f"[System]: {str(msg.content)}"))
        return trace

    def check_alignment(self, messages: list[BaseMessage]) -> AlignmentCheckResult:
        """Check alignment of the conversation trace.

        Args:
            messages: List of LangChain messages representing the conversation.

        Returns:
            AlignmentCheckResult with decision and details.
        """
        trace = self.convert_langchain_messages(messages)

        if not trace:
            return AlignmentCheckResult(
                is_misaligned=False,
                decision="ALLOW",
                reason="Empty trace",
                score=0.0
            )

        result = self.firewall.scan_replay(trace)

        # Extract decision info
        decision = str(result.decision) if hasattr(result, 'decision') else "ALLOW"
        reason = str(result.reason) if hasattr(result, 'reason') else ""
        score = float(result.score) if hasattr(result, 'score') else 0.0

        # Determine if misaligned (BLOCK or HUMAN_IN_THE_LOOP_REQUIRED)
        is_misaligned = decision != "ALLOW" and "ALLOW" not in decision

        return AlignmentCheckResult(
            is_misaligned=is_misaligned,
            decision=decision,
            reason=reason,
            score=score
        )
