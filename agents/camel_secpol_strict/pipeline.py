"""CaMeL agent with strict security policies enabled."""

from typing import Any

from agentdojo import agent_pipeline

from agents.camel.interpreter.interpreter import MetadataEvalMode
from agents.camel.pipeline import create_agent as _create_camel_agent


def create_agent(
    model: str = "gpt-4o-2024-08-06",
    **kwargs: Any,
) -> agent_pipeline.AgentPipeline:
    """Create CaMeL agent with security policies enabled (STRICT mode).

    Args:
        model: Model name with provider prefix (e.g., "openai:gpt-4o", "anthropic:claude-3-5-sonnet")
        **kwargs: Additional arguments passed to CaMeL create_agent

    Returns:
        AgentPipeline configured with CaMeL defense and strict security policies
    """
    kwargs.setdefault("use_security_policies", True)
    kwargs.setdefault("eval_mode", MetadataEvalMode.STRICT)
    return _create_camel_agent(model=model, **kwargs)
