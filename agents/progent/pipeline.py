"""Progent agent entry point for AgentDojo.

Progent is a security policy framework that uses LLM-driven policy generation
to enforce fine-grained access control on agent tool calls.
"""

import os
from typing import Any

import anthropic
import openai
from google import genai

from agentdojo.agent_pipeline import (
    AgentPipeline,
    BasePipelineElement,
    InitQuery,
    SystemMessage,
    ToolsExecutionLoop,
)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message

from agentdojo.logging import Logger

from agents.progent.secagent import (
    update_available_tools,
    generate_security_policy,
    reset_security_policy,
    set_policy_model,
    get_current_config,
    Tool,
)
from agents.progent.secagent_executor import SecagentToolsExecutor


def _add_provider_prefix(model: str) -> str:
    """Add provider prefix to model name if not present."""
    if ":" in model:
        return model

    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return f"openai:{model}"
    elif "claude" in model.lower():
        return f"anthropic:{model}"
    elif "gemini" in model.lower():
        return f"google:{model}"
    else:
        return f"openai:{model}"


def _is_oai_reasoning_model(model: str) -> bool:
    return "o4" in model or "o3" in model or "o1" in model or "codex" in model


def _get_llm(model: str, reasoning_effort: str = "medium") -> BasePipelineElement:
    """Create an LLM pipeline element for the given model."""
    from agentdojo.agent_pipeline import OpenAILLM, AnthropicLLM, GoogleLLM

    model_with_prefix = _add_provider_prefix(model)
    provider, model_name = model_with_prefix.split(":", 1)

    if provider == "google":
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        if model_name == "gemini-2.0-flash-lite-001":
            max_tokens = 8192
        else:
            max_tokens = 65535
        llm = GoogleLLM(model_name, client, max_tokens=max_tokens)
    elif provider == "openai":
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if _is_oai_reasoning_model(model_name):
            llm = OpenAILLM(client, model_name, reasoning_effort, None)
        else:
            llm = OpenAILLM(client, model_name, None)
    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        llm = AnthropicLLM(client, model_name, max_tokens=8192)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    llm.name = model_name
    return llm


def _convert_tools_to_secagent_format(tools: list) -> list[Tool]:
    """Convert AgentDojo Function objects to secagent Tool format."""
    secagent_tools = []
    for tool in tools:
        # Get JSON schema from the Pydantic model
        schema = tool.parameters.model_json_schema()
        # Extract properties from the schema
        properties = schema.get("properties", {})

        secagent_tools.append(Tool(
            name=tool.name,
            description=tool.description,
            args=properties,
        ))
    return secagent_tools


class PolicyGeneratorElement(BasePipelineElement):
    """Pipeline element that generates security policy from the user query."""

    def __init__(self, generate_policy: bool = True):
        self.generate_policy = generate_policy

    def query(
        self,
        query: str,
        runtime,
        env=None,
        messages=None,
        extra_args=None,
    ):
        if messages is None:
            messages = []
        if extra_args is None:
            extra_args = {}

        # Generate policy for the user query
        # Note: reset_security_policy() is called in benchmark.py before each task
        if self.generate_policy:
            generate_security_policy(query, manual_check=False)

            # Log the generated policy to the result JSON
            logger = Logger.get()
            if hasattr(logger, 'set_contextarg'):
                logger.set_contextarg("security_policy", get_current_config())

        return query, runtime, env, messages, extra_args


def create_agent(
    model: str = "gpt-4o-2024-08-06",
    **kwargs: Any,
) -> BasePipelineElement:
    """Create Progent agent for AgentDojo.

    Args:
        model: LLM model name (e.g., "gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022")
        **kwargs: Additional arguments:
            - suite_name: Task suite name (workspace, banking, travel, slack)
            - tools: List of AgentDojo Function objects
            - generate_policy: Whether to auto-generate policies (default: True)
            - update_policy: Whether to update policies after tool execution (default: True)
            - reasoning_effort: OpenAI reasoning effort (default: "medium")

    Returns:
        AgentPipeline configured with Progent security enforcement
    """
    suite_name = kwargs.get("suite_name", "workspace")
    tools = kwargs.get("tools", [])
    generate_policy_flag = kwargs.get("generate_policy", True)
    update_policy = kwargs.get("update_policy", True)
    reasoning_effort = kwargs.get("reasoning_effort", "medium")

    # Use the same model for policy generation as the main LLM
    model_with_prefix = _add_provider_prefix(model)
    _, model_name = model_with_prefix.split(":", 1)
    set_policy_model(model_name)

    # Register tools with secagent
    if tools:
        secagent_tools = _convert_tools_to_secagent_format(tools)
        update_available_tools(secagent_tools)

    # Create LLM
    llm = _get_llm(model, reasoning_effort)

    # Create pipeline components
    system_message = SystemMessage(load_system_message(None))
    init_query = InitQuery()
    policy_generator = PolicyGeneratorElement(generate_policy=generate_policy_flag)
    secagent_executor = SecagentToolsExecutor(update_policies=update_policy)
    tools_loop = ToolsExecutionLoop([secagent_executor, llm])

    # Build pipeline
    pipeline = AgentPipeline([
        system_message,
        init_query,
        policy_generator,
        llm,
        tools_loop,
    ])
    pipeline.name = f"progent-{model}"

    return pipeline
