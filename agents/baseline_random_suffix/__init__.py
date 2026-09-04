"""Baseline ReAct agent for AgentDojo.

This package provides a baseline LangGraph agent implementation that uses
the ReAct (Reasoning + Acting) pattern for task execution in AgentDojo.
"""

from .graph import create_agent

__all__ = ["create_agent"]
