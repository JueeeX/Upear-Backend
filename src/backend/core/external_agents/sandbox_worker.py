"""Subprocess worker entrypoint for sandboxed external-agent runs (P3.2/P3.3).

Runs INSIDE the spawned child (``python -m core.external_agents.sandbox_worker``).
Protocol: one JSON request on stdin, one JSON response written to the result
file the parent passed as argv[1], exit 0 on a clean protocol round-trip (even
when the framework run failed — ``failed: true`` carries the error). The child
environment is an allowlist plus the policy's explicit passthrough, so the
model api_key arrives via stdin (memory only, never env, never disk) and the
parent's full environment never leaks in.

Dispatch (P3.3): ``request["framework"]`` selects the execution body —
"openai-agents" / "autogen" call their runner functions from ``runners.py``
directly (the same code the in-process path runs: identical supervision
semantics, identical instructions/memory/gateway-tool assembly), anything else
keeps the legacy smolagents path byte-for-byte. The gateway tool bundle is
rebuilt from the (already filtered) child environment via the same
``build_gateway_tool_bundle`` factory the parent uses — that layer is
environment-driven by construction.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

# Frameworks whose runner functions are invoked directly in the child; every
# other request (no framework field, "smolagents", unknown) takes the legacy
# smolagents path below.
_AGENT_FRAMEWORKS = ("openai-agents", "autogen")


def _emit(payload: dict[str, Any]) -> None:
    # Result travels via the temp file the parent passed as argv[1] — stdout
    # stays free for framework library noise (temp-file-stdio discipline).
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))


def _supervision_from_request(request: dict[str, Any]) -> Any:
    from core.external_agents.supervisor import SupervisionConfig, SupervisionState

    return SupervisionState(
        config=SupervisionConfig(**{
            key: int(value)
            for key, value in (request.get("supervision") or {}).items()
        })
    )


async def _run_smolagents(request: dict[str, Any]) -> dict[str, Any]:
    from core.external_agents.gateway_tools import build_gateway_tool_bundle
    from core.external_agents.runners import (
        _step_text,
        run_code_agent_sync,
        smolagents_task,
    )

    supervision = _supervision_from_request(request)

    def _on_step(step: Any, **_kwargs: Any) -> None:
        intervention = supervision.check(output_text=_step_text(step))
        if intervention is not None:
            from core.external_agents.runners import SupervisionHalt

            raise SupervisionHalt(intervention)

    started = time.monotonic()
    halted = False
    task = smolagents_task(
        request.get("history") or [],
        str(request.get("message") or ""),
        str(request.get("long_term") or ""),
    )
    bundle = await build_gateway_tool_bundle(
        session_id=str(request.get("run_id") or ""),
        run_id=str(request.get("run_id") or ""),
        environ=dict(os_environ()),
    )
    try:
        reply = run_code_agent_sync(
            task,
            base_url=str(request.get("base_url") or ""),
            api_key=str(request.get("api_key") or ""),
            model=str(request.get("model") or ""),
            supervision=supervision,
            step_callbacks=[_on_step],
            gateway_bundle=bundle,
        )
    except Exception as exc:  # noqa: BLE001 — the child must always answer
        from core.external_agents.runners import SupervisionHalt

        if isinstance(exc, SupervisionHalt):
            supervision.interventions.append(exc.intervention)
            halted = True
            reply = ""
        else:
            return {
                "ok": True,
                "failed": True,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "duration_ms": int((time.monotonic() - started) * 1000),
                "steps": supervision.steps,
            }
    duration_ms = int((time.monotonic() - started) * 1000)
    final_check = supervision.check(output_text=reply, final=True)
    if final_check is not None:
        halted = True
        reply = ""
    return {
        "ok": True,
        "reply": reply,
        "steps": supervision.steps,
        "interventions": list(supervision.interventions),
        "halted": halted,
        "duration_ms": duration_ms,
    }


async def _run_runner_framework(request: dict[str, Any], framework: str) -> dict[str, Any]:
    """openai-agents / autogen inside the child: run the very runner function
    the in-process path uses. The runner owns the supervision semantics (final
    check, halt handling, model-client cleanup); any exception answers as a
    ``failed`` payload so the parent keeps its fail-closed audit flow instead
    of losing the run silently."""
    from core.external_agents.gateway_tools import build_gateway_tool_bundle
    from core.external_agents.runners import RUNNERS

    supervision = _supervision_from_request(request)
    bundle = await build_gateway_tool_bundle(
        session_id=str(request.get("run_id") or ""),
        run_id=str(request.get("run_id") or ""),
        environ=dict(os_environ()),
    )
    # Retry-pressure feed (supervision trigger 5), same wiring as the parent's
    # registry: only when the operator enabled it and a bundle was rebuilt.
    if bundle is not None and supervision.config.retry_trigger > 0:
        bundle["on_denial"] = supervision.record_tool_denial
    runner = RUNNERS.get(framework)
    if runner is None:
        return {
            "ok": False,
            "error_type": "ProtocolError",
            "error": f"worker has no runner for framework: {framework}",
        }
    config: dict[str, Any] = {
        "base_url": str(request.get("base_url") or ""),
        "api_key": str(request.get("api_key") or ""),
        "model": str(request.get("model") or ""),
        "history": list(request.get("history") or []),
        "long_term": str(request.get("long_term") or ""),
        "supervision": supervision,
        "gateway_tool": bundle,
        "run_id": str(request.get("run_id") or ""),
    }
    started = time.monotonic()
    try:
        result = await runner(str(request.get("message") or ""), config)
    except Exception as exc:  # noqa: BLE001 — the child must always answer
        return {
            "ok": True,
            "failed": True,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "duration_ms": int((time.monotonic() - started) * 1000),
            "steps": supervision.steps,
        }
    return {
        "ok": True,
        "reply": result.reply,
        "steps": int(result.steps or 0),
        "interventions": list(result.interventions),
        "halted": bool(result.halted),
        "duration_ms": int(result.duration_ms or 0),
    }


async def _run(request: dict[str, Any]) -> dict[str, Any]:
    framework = str(request.get("framework") or "")
    if framework in _AGENT_FRAMEWORKS:
        return await _run_runner_framework(request, framework)
    return await _run_smolagents(request)


def os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def main() -> int:
    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 — protocol errors answer, not crash
        _emit({"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]})
        return 1
    try:
        _emit(asyncio.run(_run(request)))
    except Exception as exc:  # noqa: BLE001
        _emit({"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
