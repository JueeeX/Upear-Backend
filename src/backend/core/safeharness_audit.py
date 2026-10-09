"""SQLite-backed SafeHarness audit sink for the product runtime.

SafeHarness emits already-redacted audit records; the in-memory sink kept them
for the lifetime of one process, so nothing survived a restart. This sink
persists the same facts into a small SQLite table so enforcement decisions can
be reviewed after the fact. Payloads are truncated defensively — the upstream
records are authoritative for redaction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_MAX_FIELD_BYTES = 16 * 1024
# Opportunistic pruning cadence: the prune itself runs on the write thread, so
# it must stay rare. First write after process start always attempts one pass.
_PRUNE_INTERVAL_SECONDS = 6 * 3600
_DEFAULT_RETENTION_DAYS = 90
_DEFAULT_MAX_ROWS = 200_000
_last_prune_monotonic: dict[str, float] = {}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS safeharness_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_type TEXT NOT NULL,
    event_kind TEXT NOT NULL,
    tenant_id TEXT,
    session_id TEXT,
    run_id TEXT,
    trace_id TEXT,
    agent_id TEXT,
    framework TEXT,
    decision_disposition TEXT,
    decision_reason_code TEXT,
    event_payload TEXT,
    details TEXT,
    occurred_at TEXT,
    written_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_safeharness_audit_session ON safeharness_audit(session_id);
CREATE INDEX IF NOT EXISTS idx_safeharness_audit_run ON safeharness_audit(run_id);
CREATE INDEX IF NOT EXISTS idx_safeharness_audit_kind ON safeharness_audit(event_kind);
CREATE INDEX IF NOT EXISTS idx_safeharness_audit_written ON safeharness_audit(written_at);
"""


def _bounded_json(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        encoded = json.dumps({"_unserializable": type(value).__name__})
    if len(encoded.encode("utf-8")) > _MAX_FIELD_BYTES:
        return encoded.encode("utf-8")[:_MAX_FIELD_BYTES].decode("utf-8", "ignore") + "…[truncated]"
    return encoded


def _first_reason_code(decision: Any) -> str | None:
    reasons = getattr(decision, "reasons", None) or ()
    if not reasons:
        return None
    code = getattr(reasons[0], "code", None)
    return str(code) if code else None


def _retention_config() -> tuple[int, int]:
    """(retention_days, max_rows); 0 disables that pruning dimension."""

    def _int_env(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    return (
        _int_env("SAFEHARNESS_AUDIT_RETENTION_DAYS", _DEFAULT_RETENTION_DAYS),
        _int_env("SAFEHARNESS_AUDIT_MAX_ROWS", _DEFAULT_MAX_ROWS),
    )


def prune_audit_database(
    path: str | os.PathLike[str],
    *,
    retention_days: int,
    max_rows: int,
) -> dict[str, int]:
    """Delete audit rows beyond the retention window and row cap.

    Age pruning compares the audit timestamp (occurred_at, falling back to
    written_at — both ISO-8601 so lexicographic comparison is correct); the row
    cap keeps the newest ``max_rows`` entries by that same timestamp. Returns
    the number of rows removed by each dimension so callers can log it.
    """

    from datetime import datetime, timedelta, timezone

    pruned = {"pruned_age": 0, "pruned_cap": 0}
    if retention_days <= 0 and max_rows <= 0:
        return pruned
    conn = sqlite3.connect(str(path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        with conn:
            if retention_days > 0:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
                cursor = conn.execute(
                    "DELETE FROM safeharness_audit WHERE COALESCE(occurred_at, written_at) < ?",
                    (cutoff,),
                )
                pruned["pruned_age"] = cursor.rowcount if cursor.rowcount > 0 else 0
            if max_rows > 0:
                count = conn.execute("SELECT COUNT(*) FROM safeharness_audit").fetchone()[0]
                if count > max_rows:
                    cursor = conn.execute(
                        "DELETE FROM safeharness_audit WHERE id NOT IN ("
                        "SELECT id FROM safeharness_audit "
                        "ORDER BY COALESCE(occurred_at, written_at) DESC, id DESC LIMIT ?)",
                        (max_rows,),
                    )
                    pruned["pruned_cap"] = cursor.rowcount if cursor.rowcount > 0 else 0
    finally:
        conn.close()
    return pruned


def _maybe_prune_sync(path: str) -> None:
    """Run the prune at most once per interval; never break the write path.

    A missing dict entry means "never pruned" — monotonic-clock values are only
    compared against other monotonic readings, never against a zero sentinel.
    """

    retention_days, max_rows = _retention_config()
    if retention_days <= 0 and max_rows <= 0:
        return
    now = time.monotonic()
    last = _last_prune_monotonic.get(path)
    if last is not None and now - last < _PRUNE_INTERVAL_SECONDS:
        return
    _last_prune_monotonic[path] = now
    try:
        pruned = prune_audit_database(path, retention_days=retention_days, max_rows=max_rows)
        if pruned["pruned_age"] or pruned["pruned_cap"]:
            logger.info(
                "[safeharness] audit pruned: age=%s cap=%s (retention=%sd max_rows=%s)",
                pruned["pruned_age"], pruned["pruned_cap"], retention_days, max_rows,
            )
    except Exception as exc:
        logger.warning("[safeharness] audit prune failed: %s", type(exc).__name__)


class SQLiteAuditSink:
    """Persist AuditRecord rows into SQLite (WAL, one short transaction each)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    async def write(self, record: Any) -> None:
        await asyncio.to_thread(self._write_sync, record)

    def _write_sync(self, record: Any) -> None:
        from datetime import datetime, timezone

        event = getattr(record, "event", None)
        meta = getattr(event, "meta", None)
        decision = getattr(record, "decision", None)
        # Audit minimisation (framework `runtime.redaction`), env-gated: with
        # SAFEHARNESS_AUDIT_REDACTION enabled, ledger payloads carry content
        # digests instead of raw text and sensitive keys are masked. Off by
        # default — unconfigured deployments keep today's payloads verbatim.
        if os.environ.get("SAFEHARNESS_AUDIT_REDACTION", "").strip().lower() in ("1", "true", "yes"):
            try:
                from safeharness.runtime.redaction import minimize_decision, minimize_event

                event = minimize_event(event)
                if decision is not None:
                    decision = minimize_decision(decision)
                    event = decision.effective_event
            except Exception:  # noqa: BLE001 — minimisation must never break persistence
                pass
        kind = getattr(getattr(event, "kind", None), "value", None) or str(getattr(event, "kind", ""))
        occurred = getattr(meta, "timestamp", None)
        details = dict(getattr(record, "details", {}) or {})
        # Enforcement receipts are the tamper-evident proof behind a tool
        # execution; without this merge the receipt never reaches the ledger
        # and an offline verifier has nothing to check.
        receipt = getattr(record, "receipt", None)
        if receipt is not None:
            details.setdefault("enforcement_receipt_record", {
                "receipt_id": str(getattr(receipt, "receipt_id", "") or ""),
                "decision_id": str(getattr(receipt, "decision_id", "") or ""),
                "disposition": str(getattr(getattr(receipt, "disposition", None), "value", "") or getattr(receipt, "disposition", "") or ""),
                "enforced": bool(getattr(receipt, "enforced", False)),
                "adapter": str(getattr(receipt, "adapter", "") or ""),
                "enforced_at": getattr(receipt, "enforced_at", None).isoformat()
                if isinstance(getattr(receipt, "enforced_at", None), datetime)
                else None,
                "details": dict(getattr(receipt, "details", {}) or {}),
            })
        row = (
            str(getattr(record, "record_type", "") or ""),
            str(kind or ""),
            getattr(meta, "tenant_id", None),
            getattr(meta, "session_id", None),
            getattr(meta, "run_id", None),
            getattr(meta, "trace_id", None),
            getattr(meta, "agent_id", None),
            getattr(meta, "framework", None),
            getattr(getattr(decision, "disposition", None), "value", None),
            _first_reason_code(decision),
            _bounded_json(getattr(event, "payload", {}) or {}),
            _bounded_json(details),
            occurred.isoformat() if isinstance(occurred, datetime) else (str(occurred) if occurred else None),
            datetime.now(timezone.utc).isoformat(),
        )
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO safeharness_audit ("
                    "record_type, event_kind, tenant_id, session_id, run_id, trace_id, agent_id,"
                    " framework, decision_disposition, decision_reason_code, event_payload, details,"
                    " occurred_at, written_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    row,
                )
        finally:
            conn.close()
        _maybe_prune_sync(self.path)


_SINK_CACHE: dict[str, SQLiteAuditSink] = {}


async def record_blocked_input(
    context: Mapping[str, str],
    reason_code: str,
    *,
    source: str = "user_input",
) -> None:
    """Persist a structural (regex-level) input block.

    Requests stopped by the pre-session pattern check never reach a SafeHarness
    session, so nothing else would write them to the audit trail. The recorded
    shape mirrors session-written decisions (redacted payload, DENY disposition)
    and reuses the policy's reason code.
    """

    try:
        from safeharness.api.decisions import Decision, Disposition, Reason
        from safeharness.api.events import EventKind, EventMeta, SecurityEvent
        from safeharness.ports import AuditRecord

        meta = EventMeta(
            tenant_id=str(context.get("tenant_id") or ""),
            session_id=str(context.get("session_id") or ""),
            run_id=str(context.get("run_id") or ""),
            agent_id=str(context.get("agent_id") or ""),
            sequence=1,
            framework="sumc-protected-chat",
            trace_id=str(context.get("trace_id") or ""),
        )
        event = SecurityEvent(
            kind=EventKind.CONTENT_RECEIVED,
            meta=meta,
            payload={"source": source, "reason_code": reason_code, "redacted": True},
        )
        decision = Decision(
            disposition=Disposition.DENY,
            effective_event=event,
            reasons=(Reason(reason_code, "Inbound content matched an instruction-injection policy.", "safeharness.inform.patterns"),),
            policy_version="safeharness.default.v1",
        )
        await build_audit_sink().write(AuditRecord("decision", event, decision=decision))
    except Exception as exc:
        logger.warning("[safeharness] blocked-input audit write failed: %s", type(exc).__name__)


def resolve_audit_db_path() -> str:
    """Explicit env override wins; otherwise sit next to the app's SQLite DB."""

    explicit = os.environ.get("SAFEHARNESS_AUDIT_DB", "").strip()
    if explicit:
        return explicit
    database_url = os.environ.get("DATABASE_URL", "").strip()
    for prefix in ("sqlite:///", "sqlite://"):
        if database_url.startswith(prefix):
            base = database_url[len(prefix):]
            if base and base != ":memory:":
                return str(Path(base).with_name("safeharness_audit.db"))
            # Longer prefix matched: do not let the shorter one re-parse the
            # remainder (":memory:" would otherwise become "/:memory:").
            break
    return ""


def build_audit_sink() -> Any:
    """Return the persistent sink when a target path is resolvable.

    Falls back to the in-memory sink (with a warning) instead of failing the
    runtime when no path is configured or the DB cannot be opened — the demo
    must stay runnable, and the fallback is visible in the log.
    """

    from safeharness.backends import InMemoryAuditSink

    path = resolve_audit_db_path()
    if not path:
        return InMemoryAuditSink()
    cached = _SINK_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        sink = SQLiteAuditSink(path)
    except Exception as exc:
        logger.warning(
            "[safeharness] audit persistence unavailable (%s); falling back to in-memory",
            type(exc).__name__,
        )
        return InMemoryAuditSink()
    _SINK_CACHE[path] = sink
    logger.info("[safeharness] audit persistence enabled: %s", path)
    return sink


__all__ = [
    "SQLiteAuditSink",
    "build_audit_sink",
    "prune_audit_database",
    "record_blocked_input",
    "resolve_audit_db_path",
]
