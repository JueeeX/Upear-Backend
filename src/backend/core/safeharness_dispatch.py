"""Deterministic platform-level routing for the protected demo chat."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


DispatchTarget = Literal["noc", "internet_search", "knowledge_qa", "general"]

_OPERATIONS = re.compile(
    r"(?:网元|告警|故障|排障|排查|OLT|丢包|链路|端口|配置|变更|重启|设备|NMS|alarm|fault|diagnos|network)",
    re.IGNORECASE,
)
_INTERNET = re.compile(r"(?:联网|网络搜索|公开资料|实时信息|最新资讯|internet|search|web)", re.IGNORECASE)
_KNOWLEDGE = re.compile(r"(?:知识库|知识问答|内部文档|知识检索|knowledge base|documentation)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DispatchDecision:
    target: DispatchTarget
    route: tuple[str, ...]
    reason: str
    dispatcher_agent_id: str = "platform-supervisor-agent"

    @property
    def terminal_agent_id(self) -> str:
        return self.route[-1]


def classify_protected_message(message: str) -> DispatchDecision:
    """Classify without a model; SafeHarness remains the first enforcement boundary."""

    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if _OPERATIONS.search(message):
        return DispatchDecision(
            "noc",
            ("platform-supervisor-agent", "noc-supervisor-agent", "noc-diagnosis-agent"),
            "operations_intent",
        )
    if _KNOWLEDGE.search(message):
        return DispatchDecision(
            "knowledge_qa",
            ("platform-supervisor-agent", "knowledge-qa-agent"),
            "knowledge_intent",
        )
    if _INTERNET.search(message):
        return DispatchDecision(
            "internet_search",
            ("platform-supervisor-agent", "internet-search-agent"),
            "internet_intent",
        )
    return DispatchDecision(
        "general",
        ("platform-supervisor-agent", "main_agent"),
        "general_intent",
    )


def detect_prompt_injection(message: str) -> str | None:
    """Use SafeHarness' structural input policy before platform routing."""

    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    from safeharness.policies.inform import InformPolicy

    _, pattern_matched, _ = InformPolicy()._sanitize_value(message)
    return "prompt_injection_detected" if pattern_matched else None


def detect_dangerous_sql(message: str) -> str | None:
    """Structural destructive-SQL check on inbound text (judge-free fast path)."""

    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    from safeharness.policies.sql_guard import inspect_sql

    verdict = inspect_sql(message)
    return verdict.reason_code if verdict.dangerous else None


__all__ = [
    "DispatchDecision",
    "DispatchTarget",
    "classify_protected_message",
    "detect_dangerous_sql",
    "detect_prompt_injection",
]
