"""Utility functions for baseline LangGraph agent."""

from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


class AttachmentItem(BaseModel):
    """Attachment item for email."""
    type: str = Field("file", description="The type of attachment, must be 'file'")
    file_id: int = Field(..., description="The unique identifier for the file")


class SendEmailInput(BaseModel):
    """Input schema for send_email tool (Gemini-specific)."""
    recipients: list[str]
    subject: str
    body: str
    attachments: List[AttachmentItem] = None
    cc: list[str] = None
    bcc: list[str] = None


class CreateCalendarEventInput(BaseModel):
    """Input schema for create_calendar_event tool (Gemini-specific)."""
    title: str
    start_time: str
    end_time: str
    description: str = ""
    participants: List[str] = None
    location: str = None


def bind_tools(
    llm,
    tools,
    enable_parallel_tool_calls: bool = False,
    tool_call_only: bool = False
):
    """Bind tools to an LLM instance with provider-specific configurations.

    Args:
        llm: The language model instance (OpenAI, Anthropic, or Google).
        tools: List of tools to bind to the model.
        enable_parallel_tool_calls: Whether to enable parallel tool calls.
        tool_call_only: Whether to force tool calls only (no text responses).

    Returns:
        LLM instance with tools bound.

    Raises:
        ValueError: If the LLM type is not supported.
    """
    if isinstance(llm, ChatOpenAI):
        tool_choice = "required" if tool_call_only else "auto"
        parallel_tool_calls = enable_parallel_tool_calls
        return llm.bind_tools(tools, parallel_tool_calls=parallel_tool_calls, tool_choice=tool_choice)
    elif isinstance(llm, ChatAnthropic):
        tool_choice = {
            "type": "any" if tool_call_only else "auto",
            "disable_parallel_tool_use": not enable_parallel_tool_calls,
        }
        return llm.bind_tools(tools, tool_choice=tool_choice)
    elif isinstance(llm, ChatGoogleGenerativeAI):
        tool_choice = "any" if tool_call_only else "auto"

        # Apply custom schemas for Gemini-specific tools
        for tool in tools:
            if tool.name == "send_email":
                tool.args_schema = SendEmailInput
            if tool.name == "create_calendar_event":
                tool.args_schema = CreateCalendarEventInput

        return llm.bind_tools(tools, tool_choice=tool_choice)
    else:
        raise ValueError("Invalid LLM type")


def deserialize_plugin_result(content: str):
    """Deserialize plugin result from PFI agent format.

    TODO: Implement deserialization logic for PFI plugin results.
    This function should parse and transform plugin output into a standardized format.

    Args:
        content: The plugin result content to deserialize.

    Returns:
        Deserialized plugin result (currently returns content as-is).
    """
    # For now, just return the content as-is
    # TODO: Implement proper deserialization when PFI agent integration is needed
    return content
