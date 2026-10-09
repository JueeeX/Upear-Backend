"""Demo-only adversarial probe endpoint for AgentProbe.

Single-turn synchronous wrapper around the SAME protected chat stream the real
chat route consumes, so probe results measure the production SafeHarness chain.
Gated by SAFEHARNESS_PROBE_KEY; without that env var the route authenticates
nobody and answers 404 to everything. Never enable in production.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.routes.v1 import safeharness as safeharness_routes
from core.infra.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/security", tags=["security-probe"])

_PROBE_LOG_ENV = "SAFEHARNESS_PROBE_LOG"


def _probe_key() -> str:
    return os.environ.get("SAFEHARNESS_PROBE_KEY", "")


@router.post("/probe-chat", summary="AgentProbe single-turn protected-chat probe (demo only)")
async def probe_chat(request: Request) -> JSONResponse:
    expected = _probe_key()
    if not expected or request.headers.get("X-Probe-Key") != expected:
        return JSONResponse(status_code=404, content={"error": "not found"})
    try:
        body = await request.json()
    except Exception:
        body = {}
    message = str((body or {}).get("message") or "")
    if not message.strip():
        return JSONResponse(status_code=400, content={"error": "message required"})

    handler = safeharness_routes._protected_chat_stream_handler
    if not callable(handler):
        return JSONResponse(status_code=503, content={"error": "protected chat unavailable"})

    # Resolve the judge/model provider exactly like the real chat route does,
    # otherwise the semantic judge never runs and every benign input is
    # fail-closed as "uncertain", which would fake a zero ASR.
    model_provider_id = ""
    try:
        from core.db.engine import SessionLocal
        from core.services.user_model_selection import resolve_user_model_provider_id

        with SessionLocal() as db:
            model_provider_id = resolve_user_model_provider_id(db, "", user_id="user_ce_admin") or ""
    except Exception as exc:
        logger.warning("[security-probe] model provider resolution failed: %s", type(exc).__name__)

    started = time.monotonic()
    context: dict[str, str] = {
        # Falls back to the gateway's default principal tenant: the probe chain
        # rides the same capability-mapping principal as the protected chat
        # (a "probe" placeholder tenant would fail gateway authorization).
        "tenant_id": os.environ.get("SAFEHARNESS_DEMO_TENANT_ID") or "fiberhome-demo",
        "user_id": "user_ce_admin",
        "role": "admin",
        "conversation_id": str((body or {}).get("conversation_id") or f"probe-{uuid.uuid4().hex[:8]}"),
        "session_id": f"probe-{uuid.uuid4().hex[:12]}",
        "agent_id": "main_agent",
        "project_id": "",
        "message": message,
        "model_provider_id": model_provider_id,
        "run_id": f"probe-{uuid.uuid4().hex[:12]}",
        "delegation_id": f"probe-{uuid.uuid4().hex[:12]}",
        "trace_id": f"probe-{uuid.uuid4().hex[:12]}",
    }

    reply_parts: list[str] = []
    blocked: Mapping[str, Any] | None = None
    finished: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None
    try:
        stream = handler(context)
        async for item in stream:
            event_type = str((item or {}).get("type") or "")
            data = item.get("data") if isinstance((item or {}).get("data"), Mapping) else {}
            if event_type == "blocked":
                blocked = data
            elif event_type == "message":
                reply_parts.append(str(data.get("content") or ""))
            elif event_type == "run_finished":
                finished = data
            elif event_type == "error":
                error = data
    except Exception as exc:
        error = {"code": "probe_stream_failed", "message": type(exc).__name__}

    duration_ms = int((time.monotonic() - started) * 1000)
    reply = "".join(reply_parts)
    if not reply and blocked is not None:
        reply = str(blocked.get("message") or "")
    status = "blocked" if blocked is not None else ("error" if error is not None else "completed")
    source = blocked or error or finished or {}
    evidence = {
        "status": status,
        "reason_code": blocked.get("reason_code") if blocked else (error or {}).get("code"),
        "final_status": (finished or {}).get("status"),
        "model_calls": (finished or {}).get("model_calls"),
        "judge_status": source.get("judge_status"),
        "judge_calls": source.get("judge_calls"),
        "judge_model": source.get("judge_model"),
        "duration_ms": duration_ms,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    log_path = os.environ.get(_PROBE_LOG_ENV, "")
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"message": message, "evidence": evidence}, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("[security-probe] evidence log write failed")
    return JSONResponse(content={"reply": reply, "evidence": evidence})
