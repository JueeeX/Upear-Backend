"""Output-side guard for external framework agents — env-gated, default OFF.

The dispatch boundary in ``registry`` scans the *input*; a clean framework
reply historically reached the user unreviewed (known gap). When
``SAFEHARNESS_EXTERNAL_OUTPUT_GUARD`` is enabled, the reply is reviewed through
the same ``semantic_review`` gate the supervision path uses — the platform
inform policy (builtin + file + runtime rules) plus the semantic judge, one
pass, fail-closed on any judge error. ``SAFEHARNESS_EXTERNAL_OUTPUT_ENFORCE``
turns a failed review into a replaced reply and skips the long-term memory
write-back; without it the guard is warn-only (scan, audit, annotate evidence,
deliver the reply unchanged) so operators can size false positives before
enforcing.

With both envs unset the code path is not entered at all: no judge call, no
audit record, no evidence key — behaviour is bit-identical to before.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ENV_OUTPUT_GUARD = "SAFEHARNESS_EXTERNAL_OUTPUT_GUARD"
ENV_OUTPUT_ENFORCE = "SAFEHARNESS_EXTERNAL_OUTPUT_ENFORCE"


def _env_flag(environ: dict[str, str] | None, name: str) -> bool:
    import os

    values = os.environ if environ is None else environ
    return (values.get(name) or "").strip().lower() in ("1", "true", "yes")


def output_guard_enabled(environ: dict[str, str] | None = None) -> bool:
    return _env_flag(environ, ENV_OUTPUT_GUARD)


def output_guard_enforce_enabled(environ: dict[str, str] | None = None) -> bool:
    return _env_flag(environ, ENV_OUTPUT_ENFORCE)


def output_notice(reason_code: str, reviewed: str) -> str:
    """The structured reply an enforce-blocked run degrades to."""
    return (
        f"输出安全守卫：该次回复未通过输出侧复查（原因：{reason_code}）。"
        f"输出安全复查：{reviewed}。可调整任务描述后重试。"
    )


async def review_output(
    reply: str,
    *,
    context: dict[str, str] | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Review a clean framework reply. ``None`` when the guard is off.

    The reply is already excluded from double-judging by the caller (the
    supervision path reviews its own target); here only the clean path lands.
    ``semantic_review`` is fail-closed internally — any judge error comes back
    as ``allowed=False``, which maps to the warn/block actions below.
    """
    if not output_guard_enabled(environ):
        return None
    enforce = output_guard_enforce_enabled(environ)
    mode = "enforce" if enforce else "warn"

    text = str(reply or "")
    if not text.strip():
        return {
            "mode": mode,
            "reason_code": "",
            "review": "skipped:empty_reply",
            "action": "skipped_empty",
            "notice": "",
        }

    from core.evolution.content_gate import semantic_review

    try:
        gate = await semantic_review(text, context=context)
    except Exception as exc:  # noqa: BLE001 — belt-and-braces: the gate is
        # fail-closed internally; this keeps a guard fault from ever failing
        # the run itself, degrading to the same not-allowed mapping.
        logger.warning("[external-agents] output review failed: %s", type(exc).__name__)
        return {
            "mode": mode,
            "reason_code": "content_inspection_unavailable",
            "review": "ERROR:content_inspection_unavailable",
            "action": "blocked" if enforce else "warn",
            "notice": output_notice("content_inspection_unavailable", "ERROR:content_inspection_unavailable")
            if enforce
            else "",
        }
    review = f"{gate.judge_status}:{gate.reason_code}"
    if gate.allowed:
        return {
            "mode": mode,
            "reason_code": str(gate.reason_code or "content_inspection_passed"),
            "review": review,
            "action": "pass",
            "notice": "",
        }
    return {
        "mode": mode,
        "reason_code": str(gate.reason_code or "semantic_prompt_injection_uncertain"),
        "review": review,
        "action": "blocked" if enforce else "warn",
        "notice": output_notice(gate.reason_code, review) if enforce else "",
    }


__all__ = [
    "ENV_OUTPUT_ENFORCE",
    "ENV_OUTPUT_GUARD",
    "output_guard_enabled",
    "output_guard_enforce_enabled",
    "output_notice",
    "review_output",
]
