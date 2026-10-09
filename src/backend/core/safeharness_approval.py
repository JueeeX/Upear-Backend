"""One-shot human approval state for SafeHarness protected chat."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass

from safeharness.ports import ApprovalResult

KIND = "safeharness"
OP = "safeharness_approval"
ALLOW = "allow"
DENY = "deny"
TIMEOUT = "timeout"


@dataclass
class _Pending:
    tool_name: str
    decision_id: str
    summary: str
    event: asyncio.Event
    created_at: float
    decision: str | None = None
    approver: str | None = None


_LOCK = threading.RLock()
_PENDING: dict[str, dict[str, _Pending]] = {}


def _info(confirm_id: str, pending: _Pending) -> dict[str, str]:
    return {
        "confirm_id": confirm_id,
        "op": OP,
        "logical_path": pending.tool_name,
        "message": pending.summary,
        "kind": KIND,
    }


def get_all_pending(chat_id: str) -> list[dict[str, str]]:
    with _LOCK:
        items = [
            (confirm_id, item)
            for confirm_id, item in _PENDING.get(chat_id, {}).items()
            if item.decision is None
        ]
        items.sort(key=lambda pair: pair[1].created_at)
        return [_info(confirm_id, item) for confirm_id, item in items]


def get_pending(chat_id: str) -> dict[str, str] | None:
    items = get_all_pending(chat_id)
    return items[-1] if items else None


def list_pending_chat_ids() -> list[str]:
    with _LOCK:
        return [
            chat_id
            for chat_id, items in _PENDING.items()
            if any(item.decision is None for item in items.values())
        ]


def set_decision(
    chat_id: str,
    confirm_id: str,
    decision: str,
    *,
    approver: str,
) -> dict[str, object]:
    with _LOCK:
        item = _PENDING.get(chat_id, {}).get(confirm_id)
        if item is None or item.decision is not None:
            return {"ok": False, "reason": "stale", "error": "approval expired"}
        if decision not in (ALLOW, DENY):
            return {
                "ok": False,
                "reason": "bad_decision",
                "error": "SafeHarness approval is one-shot only",
            }
        item.decision = decision
        item.approver = str(approver).strip() or None
        item.event.set()
        return {"ok": True, "decision": decision, "approver": item.approver}


async def approval_gate(
    *,
    chat_id: str | None,
    tool_name: str,
    decision_id: str,
    timeout: float = 7200,
) -> dict[str, object]:
    if not chat_id or not tool_name or not decision_id:
        return {"approved": False, "reason": "approval_context_unavailable"}

    confirm_id = uuid.uuid4().hex[:12]
    item = _Pending(
        tool_name,
        decision_id,
        f"允许 SafeHarness 工具访问：{tool_name}",
        asyncio.Event(),
        time.monotonic(),
    )
    with _LOCK:
        _PENDING.setdefault(chat_id, {})[confirm_id] = item
    try:
        await asyncio.wait_for(item.event.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        item.decision = TIMEOUT
    finally:
        with _LOCK:
            _PENDING.get(chat_id, {}).pop(confirm_id, None)
            if not _PENDING.get(chat_id):
                _PENDING.pop(chat_id, None)

    if item.decision == ALLOW and item.approver:
        return {
            "approved": True,
            "approval_id": confirm_id,
            "approver": item.approver,
        }
    return {
        "approved": False,
        "approval_id": confirm_id,
        "reason": item.decision or TIMEOUT,
    }


class SUMCApprovalBroker:
    async def request(self, context, event, decision) -> ApprovalResult:
        result = await approval_gate(
            chat_id=str(context.metadata.get("conversation_id") or ""),
            tool_name=str(event.payload.get("tool_name") or ""),
            decision_id=decision.decision_id,
        )
        return ApprovalResult(
            approved=bool(result.get("approved")),
            approver=str(result.get("approver") or ""),
            approval_id=str(result.get("approval_id") or ""),
            reason=str(result.get("reason") or ""),
        )
