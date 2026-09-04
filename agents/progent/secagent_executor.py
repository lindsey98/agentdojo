"""SecagentToolsExecutor - Tool executor with security policy enforcement."""

from ast import literal_eval
from collections.abc import Callable, Sequence
import json

from pydantic import BaseModel

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionReturnType, FunctionsRuntime
from agentdojo.types import ChatMessage, ChatToolResultMessage, text_content_block_from_string

from agents.progent.secagent import (
    check_tool_call,
    generate_update_security_policy,
    ValidationError,
)


def is_string_list(s: str):
    try:
        parsed = literal_eval(s)
        return isinstance(parsed, list)
    except (ValueError, SyntaxError):
        return False


def tool_result_to_str(
    tool_result: FunctionReturnType, dump_fn: Callable[[dict | list[dict]], str] = json.dumps
) -> str:
    """Basic tool output formatter with JSON dump by default."""
    if isinstance(tool_result, BaseModel):
        return dump_fn(tool_result.model_dump(mode='json')).strip()

    if isinstance(tool_result, list):
        res_items = []
        for item in tool_result:
            if type(item) in [str, int]:
                res_items += [str(item)]
            elif isinstance(item, BaseModel):
                res_items += [item.model_dump(mode='json')]
            else:
                raise TypeError("Not valid type for item tool result: " + str(type(item)))
        return dump_fn(res_items).strip()

    if isinstance(tool_result, dict):
        return dump_fn(tool_result).strip()

    return str(tool_result)


class SecagentToolsExecutor(BasePipelineElement):
    """Tool executor with Progent secagent security policy enforcement.

    Before executing each tool call, this executor checks the tool call against
    the security policy using secagent's check_tool_call function. If the check
    fails (ValidationError), the tool call is not executed and an error message
    is returned to the LLM.

    Args:
        tool_output_formatter: a function that converts a tool's output into plain text.
        update_policies: if True, call generate_update_security_policy after each tool execution
            to potentially update the security policy based on tool results.
    """

    def __init__(
        self,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        update_policies: bool = True,
    ) -> None:
        self.output_formatter = tool_output_formatter
        self.update_policies = update_policies

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args
        if messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        if messages[-1]["tool_calls"] is None or len(messages[-1]["tool_calls"]) == 0:
            return query, runtime, env, messages, extra_args

        tool_call_results = []
        for tool_call in messages[-1]["tool_calls"]:
            if tool_call.function == EMPTY_FUNCTION_NAME:
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error="Empty function name provided. Provide a valid function name.",
                    )
                )
                continue
            if tool_call.function not in (tool.name for tool in runtime.functions.values()):
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string("")],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=f"Invalid tool {tool_call.function} provided.",
                    )
                )
                continue

            # Converts type of input lists from string to list type
            for arg_k, arg_v in tool_call.args.items():
                if isinstance(arg_v, str) and is_string_list(arg_v):
                    tool_call.args[arg_k] = literal_eval(arg_v)

            # Check security policy before executing tool
            try:
                check_tool_call(tool_call.function, tool_call.args)
            except ValidationError as e:
                # Policy check failed - return error message instead of executing
                tool_call_results.append(
                    ChatToolResultMessage(
                        role="tool",
                        content=[text_content_block_from_string(str(e))],
                        tool_call_id=tool_call.id,
                        tool_call=tool_call,
                        error=str(e),
                    )
                )
                continue

            # Policy check passed - execute the tool
            tool_call_result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            tool_call_id = tool_call.id
            formatted_tool_call_result = self.output_formatter(tool_call_result)

            # Optionally update security policy based on tool result
            if self.update_policies:
                try:
                    generate_update_security_policy(
                        {"name": tool_call.function, "args": tool_call.args},
                        formatted_tool_call_result,
                        manual_check=False,  # Don't prompt for confirmation in benchmark
                    )
                except Exception:
                    # Ignore policy update errors - they shouldn't block tool execution
                    pass

            tool_call_results.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(formatted_tool_call_result)],
                    tool_call_id=tool_call_id,
                    tool_call=tool_call,
                    error=error,
                )
            )
        return query, runtime, env, [*messages, *tool_call_results], extra_args
