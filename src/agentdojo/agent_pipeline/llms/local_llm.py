"""Use a locally hosted model.

Expects an OpenAI-compatible API server to be running on port 8000, e.g. launched with:

```
vllm serve /path/to/huggingface/model
```
"""

import ast
import json
import random
import re
from collections.abc import Collection, Sequence

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.logging import record_token_usage
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string


class InvalidModelOutputError(Exception): ...


def reformat_message(message: ChatCompletionMessageParam):
    if message["role"] == "user" or message["role"] == "assistant":
        content = ""
        if "content" in message and message["content"] is not None:
            for message_content in message["content"]:
                if isinstance(message_content, str):
                    content += message_content
                elif "content" in message_content:
                    content += message_content["content"]
                else:
                    content += str(message_content)
                content += "\n\n"
            content = content.strip()
    else:
        content = message["content"]
    return content


def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float | None = 1.0,
    top_p: float | None = 0.9,
) -> str:
    try:
        reformatted_messages = []
        for message in messages:
            content = reformat_message(message)
            reformatted_messages.append({"role": message["role"], "content": content})
        completion = client.chat.completions.create(
            model=model,
            messages=reformatted_messages,
            temperature=temperature,
            top_p=top_p,
            seed=random.randint(0, 1000000),
        )
        record_token_usage(getattr(completion, "usage", None))
        response = completion.choices[0].message.content
    except Exception as e:
        print(f"[debug] error: {e}")
        response = ""
    if response is None:
        raise InvalidModelOutputError("No response from model")
    return response


_tool_calling_prompt = """# Instructions
You are a helpful assistant. You will be given a task and have access to a set of possible functions which you may call in order to generate a final answer to the question. Functions must be called one at a time, but you may continue to call additional functions if you need to before providing your final answer.

## Currently Available Functions
{functions}

## Function Calling
You may output your reasoning prior to calling a function.

If you choose to call a particular function, include the function call in the following format as part of your response:

```
<function=function_name>{{"param1": "val1", ...}}</function>
```

where the JSON inside <function=...> contains exactly the parameters for that function. Pass the arguments in correct format, i.e., strings should be enclosed in quotes, lists should be enclosed in square brackets, integers should have no quotes, etc. If there are no parameters, then you should still include an empty JSON object `{{}}` inside the <function=...> tag.

### Function Results
After calling a function, stop generating text and wait for the results of the function call in the next message. The next message will use provide the output of the function call as a JSON object. Based on the output of the function call, you may call additional functions or provide a final answer."""


def _make_system_prompt(system_message: str, tools: Collection[Function]) -> str:
    if len(tools) == 0:
        return system_message

    functions = ""
    for tool in tools:
        tool_dict = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        }
        functions += json.dumps(tool_dict, indent=2)
        functions += "\n\n"

    prompt = _tool_calling_prompt.format(functions=functions)

    if system_message:
        prompt += "\n\n" + "## Additional Instructions" + "\n\n" + system_message

    return prompt


# Tool-call formats emitted by prompted local models. Both are anchored on their closing tag so
# the argument block (`.*?`, non-greedy) still spans nested braces:
#   <function=NAME>{...}</function>          (Qwen-style, prompted by _tool_calling_prompt)
#   <|tool_call>call:NAME{...}<tool_call|>   (Gemma-style)
_FUNCTION_TAG_RE = re.compile(r"<function\s*=\s*([^>]+?)>(.*?)</function>", re.S)
_GEMMA_TOOL_CALL_RE = re.compile(r"<\|tool_call>\s*call:\s*([A-Za-z0-9_]+)\s*(\{.*?\})\s*<tool_call\|>", re.S)
# Unterminated `<function=NAME>{...}` with no closing tag (fallback for a truncated generation).
_FUNCTION_OPEN_RE = re.compile(r"<function\s*=\s*([^>]+)>")


def _loads_args(raw: str) -> dict | None:
    """Parse a tool-call argument block into a dict, tolerantly. Handles valid JSON, an empty body
    (no-arg call), and the unquoted-key pseudo-JSON some models emit (e.g. `{day: "2024-05-15"}`)."""
    raw = raw.strip()
    if raw == "":
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    # Quote bare object keys: {day: "x", n: 1} -> {"day": "x", "n": 1}
    fixed = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', raw)
    try:
        v = json.loads(fixed)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    try:
        v = ast.literal_eval(raw)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _parse_model_output(completion: str) -> ChatAssistantMessage:
    """Parse a prompted model's completion into an assistant message, extracting every tool call.

    Recognizes both the `<function=NAME>{...}</function>` (Qwen-style) and
    `<|tool_call>call:NAME{...}<tool_call|>` (Gemma-style) formats, supports multiple calls per
    message, and parses arguments tolerantly (see `_loads_args`)."""
    default_message = ChatAssistantMessage(
        role="assistant", content=[text_content_block_from_string(completion.strip())], tool_calls=[]
    )

    tool_calls: list[FunctionCall] = []
    for pattern in (_GEMMA_TOOL_CALL_RE, _FUNCTION_TAG_RE):
        for name, raw in pattern.findall(completion):
            args = _loads_args(raw)
            if args is None:
                continue
            try:
                tool_calls.append(FunctionCall(function=name.strip(), args=args))
            except ValidationError:
                continue

    # Fallback: a single unterminated `<function=NAME>{...}` (no closing tag) from a truncated output.
    if not tool_calls:
        m = _FUNCTION_OPEN_RE.search(completion)
        if m and "</function>" not in completion:
            raw = completion[m.end() :]
            args = _loads_args(raw)
            if args is not None:
                try:
                    tool_calls.append(FunctionCall(function=m.group(1).strip(), args=args))
                except ValidationError:
                    pass

    if not tool_calls:
        return default_message
    return ChatAssistantMessage(
        role="assistant", content=[text_content_block_from_string(completion.strip())], tool_calls=tool_calls
    )


class LocalLLM(BasePipelineElement):
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        temperature: float | None = 0.0,
        top_p: float | None = 0.9,
        tool_delimiter: str | None = "tool",
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.tool_delimiter = tool_delimiter

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        messages_ = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system" and content is not None:
                content = _make_system_prompt(get_text_content_as_str(content), runtime.functions.values())
            if role == "tool":
                role = self.tool_delimiter
                if "error" in m and m["error"] is not None:
                    content = json.dumps({"error": m["error"]})
                else:
                    func_result = m["content"]
                    if func_result == "None":
                        func_result = "Success"
                    content = json.dumps({"result": func_result})
            messages_.append({"role": role, "content": content})

        completion = chat_completion_request(
            self.client,
            model=self.model,
            messages=messages_,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        output = _parse_model_output(completion)
        return query, runtime, env, [*messages, output], extra_args
