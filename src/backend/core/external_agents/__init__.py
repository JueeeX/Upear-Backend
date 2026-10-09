"""External agent bridge — real third-party agent frameworks, guarded.

Each runner executes the upstream project's own agent loop (not a re-skin of
its prompt): smolagents (huggingface/smolagents) CodeAgent, openai-agents
(openai/openai-agents-python) ReAct-style Agent, and AutoGen (microsoft/autogen)
AssistantAgent. The model is the platform-configured main provider (the relay),
so no new credentials are introduced.

Security posture mirrors the protected-chat chain: every inbound message passes
the SafeHarness structural checks before a framework sees it (fail-closed,
audited), and every run is audited with its framework/model/latency evidence.
"""

from core.external_agents.registry import (
    available_external_agents,
    run_external_agent,
)

__all__ = ["available_external_agents", "run_external_agent"]
