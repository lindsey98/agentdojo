"""Llama Prompt Guard protected agent for AgentDojo.

This agent extends the baseline ReAct agent with Llama Prompt Guard 2
protection for detecting prompt injection attacks in tool results.
"""

from .graph import create_agent

__all__ = ["create_agent"]
