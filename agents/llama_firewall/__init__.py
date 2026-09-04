"""LlamaFirewall AlignmentCheck protected agent for AgentDojo.

This module provides a baseline agent protected by LlamaFirewall's
AlignmentCheck scanner for detecting misaligned agent behavior.
"""

from .graph import create_agent

__all__ = ["create_agent"]
