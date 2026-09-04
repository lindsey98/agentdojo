"""CaMeL agent entry point for agentdojo."""
from typing import Any

from agentdojo import agent_pipeline

from agents.camel.interpreter.interpreter import MetadataEvalMode
from agents.camel.utils.models import make_tools_pipeline


def _add_provider_prefix(model: str) -> str:
    """Add provider prefix to model name if not present.

    Args:
        model: Model name, optionally with provider prefix

    Returns:
        Model name with provider prefix (e.g., "openai:gpt-4o-mini-2024-07-18")
    """
    # Already has provider prefix
    if ":" in model:
        return model

    # Infer provider from model name
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return f"openai:{model}"
    elif "claude" in model.lower():
        return f"anthropic:{model}"
    elif "gemini" in model.lower():
        return f"google:{model}"
    else:
        # Default to OpenAI
        return f"openai:{model}"


def create_agent(
    model: str = "gpt-4o-2024-08-06",
    **kwargs: Any,
) -> agent_pipeline.AgentPipeline:
    """Create CaMeL agent for AgentDojo.

    Args:
        model: Model name with provider prefix (e.g., "openai:gpt-4o", "anthropic:claude-3-5-sonnet")
        **kwargs: Additional arguments:
            - suite_name: Task suite name (workspace, banking, travel, slack)
            - use_original: Use original tool calling without CaMeL (default: False)
            - replay_with_policies: Replay mode with security policies (default: False)
            - attack_name: Attack name for replay mode (default: "")
            - reasoning_effort: OpenAI reasoning effort (default: "medium")
            - thinking_budget_tokens: Anthropic thinking budget (default: None)
            - ad_defense: AgentDojo defense name (default: None)
            - eval_mode: Evaluation mode (default: NORMAL)
            - q_llm: Quarantine LLM model (default: None, uses main model)
            - use_security_policies: Enable suite-specific security policies (default: False)

    Returns:
        AgentPipeline configured with CaMeL defense
    """
    # Extract parameters from kwargs with defaults
    suite_name = kwargs.get("suite_name", "workspace")
    use_original = kwargs.get("use_original", False)
    replay_with_policies = kwargs.get("replay_with_policies", False)
    attack_name = kwargs.get("attack_name", "")
    reasoning_effort = kwargs.get("reasoning_effort", "medium")
    thinking_budget_tokens = kwargs.get("thinking_budget_tokens", None)
    ad_defense = kwargs.get("ad_defense", None)
    eval_mode = kwargs.get("eval_mode", MetadataEvalMode.NORMAL)
    q_llm = kwargs.get("q_llm", None)
    use_security_policies = kwargs.get("use_security_policies", False)

    # Add provider prefix if not present
    model_with_prefix = _add_provider_prefix(model)

    # Create pipeline using existing function
    pipeline = make_tools_pipeline(
        model=model_with_prefix,
        use_original=use_original,
        replay_with_policies=replay_with_policies,
        attack_name=attack_name,
        reasoning_effort=reasoning_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        suite=suite_name,
        ad_defense=ad_defense,
        eval_mode=eval_mode,
        q_llm=q_llm,
        use_security_policies=use_security_policies,
    )

    return pipeline
