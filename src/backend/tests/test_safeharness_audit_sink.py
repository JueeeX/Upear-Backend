"""SQLite audit sink: persistence, truncation, path resolution, fallback."""

import asyncio
import json
import sqlite3

from core.safeharness_audit import (
    SQLiteAuditSink,
    build_audit_sink,
    resolve_audit_db_path,
)
from safeharness.api.decisions import Decision, Disposition, Reason
from safeharness.api.events import EventKind, EventMeta, SecurityEvent
from safeharness.ports import AuditRecord


def _record(payload=None, kind: EventKind = EventKind.CONTENT_RECEIVED):
    meta = EventMeta(
        tenant_id="tenant-a",
        session_id="sess-1",
        run_id="run-1",
        agent_id="main_agent",
        sequence=1,
        framework="sumc-protected-chat",
        trace_id="trace-1",
    )
    event = SecurityEvent(kind=kind, meta=meta, payload=payload or {"content": "hello"})
    decision = Decision(
        disposition=Disposition.DENY,
        effective_event=event,
        reasons=(Reason(code="semantic_prompt_injection_detected", message="blocked"),),
    )
    return AuditRecord(record_type="decision", event=event, decision=decision, details={"stage": "input_guard"})


def test_sink_persists_decision_row(tmp_path):
    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    asyncio.run(sink.write(_record()))

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT record_type, event_kind, tenant_id, session_id, run_id, agent_id, framework, decision_disposition, decision_reason_code FROM safeharness_audit").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "decision"
    assert row[1] == "content_received"
    assert row[2] == "tenant-a"
    assert row[3] == "sess-1"
    assert row[4] == "run-1"
    assert row[5] == "main_agent"
    assert row[6] == "sumc-protected-chat"
    assert row[7] == "deny"
    assert row[8] == "semantic_prompt_injection_detected"


def test_sink_truncates_oversized_payload(tmp_path):
    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    asyncio.run(sink.write(_record(payload={"content": "x" * 40000})))

    conn = sqlite3.connect(db)
    try:
        stored = conn.execute("SELECT event_payload FROM safeharness_audit").fetchone()[0]
    finally:
        conn.close()

    assert stored.endswith("[truncated]")
    assert len(stored.encode("utf-8")) <= 16 * 1024 + 32


def test_sink_accepts_wrapped_record_shape(tmp_path):
    """The runtime hands the sink a record whose payload is not JSON-native."""
    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    record = _record(payload={"nested": {"value": object()}})

    asyncio.run(sink.write(record))

    conn = sqlite3.connect(db)
    try:
        stored = json.loads(conn.execute("SELECT event_payload FROM safeharness_audit").fetchone()[0])
    finally:
        conn.close()
    assert "nested" in stored


def test_resolve_path_prefers_explicit_env(tmp_path, monkeypatch):
    target = tmp_path / "explicit" / "custom.db"
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", str(target))
    monkeypatch.setenv("DATABASE_URL", "sqlite:///C:/data/sumc.db")

    assert resolve_audit_db_path() == str(target)


def test_resolve_path_derives_from_sqlite_database_url(tmp_path, monkeypatch):
    monkeypatch.delenv("SAFEHARNESS_AUDIT_DB", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'sumc.db'}")

    resolved = resolve_audit_db_path()

    assert resolved == str(tmp_path / "safeharness_audit.db")


def test_resolve_path_empty_for_non_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("SAFEHARNESS_AUDIT_DB", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user@localhost/sumc")

    assert resolve_audit_db_path() == ""


def test_build_audit_sink_falls_back_to_memory_without_target(monkeypatch):
    monkeypatch.delenv("SAFEHARNESS_AUDIT_DB", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user@localhost/sumc")

    from safeharness.backends import InMemoryAuditSink

    assert isinstance(build_audit_sink(), InMemoryAuditSink)


def test_build_audit_sink_caches_per_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", str(tmp_path / "audit.db"))

    first = build_audit_sink()
    second = build_audit_sink()

    assert isinstance(first, SQLiteAuditSink)
    assert first is second


def test_record_blocked_input_writes_deny_row(tmp_path, monkeypatch):
    """Regex-level blocks never reach a session; the helper must still persist them."""
    import sqlite3

    from core import safeharness_audit as audit_mod

    db = tmp_path / "audit.db"
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", str(db))
    audit_mod._SINK_CACHE.clear()

    asyncio.run(audit_mod.record_blocked_input(
        {"tenant_id": "t1", "session_id": "s1", "run_id": "r1", "agent_id": "main_agent", "trace_id": "tr1"},
        "prompt_injection_detected",
    ))

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT record_type, event_kind, agent_id, decision_disposition, decision_reason_code, event_payload"
            " FROM safeharness_audit"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    record_type, kind, agent_id, disposition, reason_code, payload = rows[0]
    assert (record_type, kind, agent_id, disposition, reason_code) == (
        "decision", "content_received", "main_agent", "deny", "prompt_injection_detected"
    )
    # No raw user content may land in the audit row.
    assert "redacted" in payload


def _seed_row(conn, *, occurred_at, record_type="decision"):
    conn.execute(
        "INSERT INTO safeharness_audit ("
        "record_type, event_kind, tenant_id, session_id, run_id, trace_id, agent_id,"
        " framework, decision_disposition, decision_reason_code, event_payload, details,"
        " occurred_at, written_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            record_type, "content_received", "tenant-a", "sess-1", "run-1", "trace-1",
            "main_agent", "sumc-protected-chat", "deny", "prompt_injection_detected",
            "{}", "{}", occurred_at, occurred_at,
        ),
    )
    conn.commit()


def test_prune_removes_only_rows_older_than_retention(tmp_path):
    from datetime import datetime, timedelta, timezone

    from core.safeharness_audit import prune_audit_database

    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    conn = sqlite3.connect(db)
    try:
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        fresh = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        _seed_row(conn, occurred_at=old)
        _seed_row(conn, occurred_at=fresh)
    finally:
        conn.close()

    pruned = prune_audit_database(str(db), retention_days=90, max_rows=0)

    assert pruned == {"pruned_age": 1, "pruned_cap": 0}
    conn = sqlite3.connect(db)
    try:
        remaining = [r[0] for r in conn.execute("SELECT occurred_at FROM safeharness_audit")]
    finally:
        conn.close()
    assert remaining == [fresh]


def test_prune_row_cap_keeps_newest(tmp_path):
    from datetime import datetime, timedelta, timezone

    from core.safeharness_audit import prune_audit_database

    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    base = datetime.now(timezone.utc)
    conn = sqlite3.connect(db)
    try:
        for days in (5, 4, 3, 2, 1):
            _seed_row(conn, occurred_at=(base - timedelta(days=days)).isoformat())
    finally:
        conn.close()

    pruned = prune_audit_database(str(db), retention_days=0, max_rows=3)

    assert pruned == {"pruned_age": 0, "pruned_cap": 2}
    conn = sqlite3.connect(db)
    try:
        remaining = [r[0] for r in conn.execute("SELECT occurred_at FROM safeharness_audit ORDER BY occurred_at")]
    finally:
        conn.close()
    assert len(remaining) == 3
    assert all(not r.startswith((base - timedelta(days=5)).isoformat()[:10]) for r in remaining)


def test_prune_disabled_when_both_limits_zero(tmp_path):
    from core.safeharness_audit import prune_audit_database

    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    conn = sqlite3.connect(db)
    try:
        _seed_row(conn, occurred_at="2000-01-01T00:00:00+00:00")
    finally:
        conn.close()

    pruned = prune_audit_database(str(db), retention_days=0, max_rows=0)

    assert pruned == {"pruned_age": 0, "pruned_cap": 0}
    conn = sqlite3.connect(db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM safeharness_audit").fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_write_path_prunes_when_due(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from core.safeharness_audit import _last_prune_monotonic

    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    conn = sqlite3.connect(db)
    try:
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        _seed_row(conn, occurred_at=old)
    finally:
        conn.close()

    monkeypatch.setenv("SAFEHARNESS_AUDIT_RETENTION_DAYS", "90")
    monkeypatch.setenv("SAFEHARNESS_AUDIT_MAX_ROWS", "0")
    monkeypatch.delitem(_last_prune_monotonic, str(db), raising=False)

    asyncio.run(sink.write(_record()))

    conn = sqlite3.connect(db)
    try:
        occurred = [r[0] for r in conn.execute("SELECT occurred_at FROM safeharness_audit")]
    finally:
        conn.close()
    assert len(occurred) == 1 and all("T" in r for r in occurred)
    assert not any(r.startswith(old[:10]) for r in occurred)

def _parse_ledger_payload(raw: str):
    """Ledger payloads are JSON-wrapped python-repr strings (the frozen mapping
    fails json encoding and default=str stringifies it) — unwrap both layers,
    the way the supervision-queue consumer does."""
    import ast

    if raw.startswith('"'):
        raw = json.loads(raw)
    if isinstance(raw, str):
        return ast.literal_eval(raw.replace("mappingproxy(", "("))
    return raw


def test_audit_redaction_off_by_default(tmp_path, monkeypatch):
    """Without the env flag the ledger keeps payloads verbatim (no change)."""
    import ast

    monkeypatch.delenv("SAFEHARNESS_AUDIT_REDACTION", raising=False)
    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    asyncio.run(sink.write(_record(payload={"content": "very secret sentence", "api_key": "k-123"})))
    raw = sqlite3.connect(db).execute(
        "SELECT event_payload FROM safeharness_audit").fetchone()[0]
    payload = _parse_ledger_payload(raw)
    assert payload["content"] == "very secret sentence"
    assert payload["api_key"] == "k-123"


def test_audit_redaction_minimizes_when_enabled(tmp_path, monkeypatch):
    """With the flag, content keys become digests and sensitive keys are masked."""
    import ast

    monkeypatch.setenv("SAFEHARNESS_AUDIT_REDACTION", "1")
    db = tmp_path / "audit.db"
    sink = SQLiteAuditSink(db)
    asyncio.run(sink.write(_record(payload={"content": "very secret sentence", "api_key": "k-123", "note": "safe"})))
    raw = sqlite3.connect(db).execute(
        "SELECT event_payload FROM safeharness_audit").fetchone()[0]
    payload = _parse_ledger_payload(raw)
    content = payload["content"]
    assert content["redacted"] is True and content["sha256"]
    assert "very secret sentence" not in json.dumps(payload)
    assert payload["api_key"] == "[REDACTED]"
    assert payload["note"] == "safe"
    # The decision row keeps its reason code after minimisation.
    reason = sqlite3.connect(db).execute(
        "SELECT decision_reason_code FROM safeharness_audit").fetchone()[0]
    assert reason == "semantic_prompt_injection_detected"
