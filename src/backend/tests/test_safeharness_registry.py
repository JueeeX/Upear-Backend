"""P1.2 implementation registry — judge provider and audit sink selection.

Default selection is bit-identical to pre-registry behaviour; unknown names
fall safe to the default (never "no judge", never "no sink"); the paranoid
judge withholds approval with zero model calls.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core import safeharness_registry as registry
from core.db.engine import Base
from core.db.models import SystemConfig


@pytest.fixture
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    s = sessionmaker(bind=eng)()
    yield s
    s.close()


def _set_selection(db, payload):
    s = sessionmaker(bind=db.bind)()
    s.add(
        SystemConfig(
            config_key=registry.IMPL_REGISTRY_CONFIG_KEY,
            config_value=json.dumps(payload),
            display_name="x",
            group_key="safeharness",
            is_secret=False,
        )
    )
    s.commit()
    s.close()


def test_default_selection_is_bit_identical(db):
    assert registry.load_selection(db) == {"judge_provider": "openai_compatible", "audit_sink": "auto"}
    assert registry.selected_judge_implementation(db) == "openai_compatible"


def test_unknown_names_fall_safe_to_default(db, caplog):
    _set_selection(db, {"judge_provider": "magic-judge", "audit_sink": "blackhole"})
    with caplog.at_level("WARNING"):
        selection = registry.load_selection(db)
    assert selection == {"judge_provider": "openai_compatible", "audit_sink": "auto"}
    assert any("magic-judge" in m for m in caplog.messages)
    assert any("blackhole" in m for m in caplog.messages)


def test_known_selection_reads_through(db):
    _set_selection(db, {"judge_provider": "paranoid", "audit_sink": "memory"})
    selection = registry.load_selection(db)
    assert selection["judge_provider"] == "paranoid"
    assert selection["audit_sink"] == "memory"


def test_malformed_config_fails_safe(db):
    s = sessionmaker(bind=db.bind)()
    s.add(
        SystemConfig(
            config_key=registry.IMPL_REGISTRY_CONFIG_KEY,
            config_value="{not json",
            display_name="x",
            group_key="safeharness",
            is_secret=False,
        )
    )
    s.commit()
    s.close()
    assert registry.load_selection(db) == {"judge_provider": "openai_compatible", "audit_sink": "auto"}


def test_paranoid_judge_uncertain_zero_calls():
    judge = registry.paranoid_judge()

    async def scenario():
        from safeharness.ports.base import JudgeRequest

        request = JudgeRequest(purpose="prompt_injection_detection", content={"content": "hello"})
        result = await judge.judge(request)
        return result

    result = asyncio.run(scenario())
    assert result.label == "UNCERTAIN"
    assert result.judge_calls == 0
    assert result.judge_status == "ok"
    assert registry.paranoid_judge() is judge  # stateless singleton


def test_audit_registry_memory_forces_memory_despite_env(db, monkeypatch):
    import tempfile

    # A resolvable sqlite path is irrelevant when the registry pins memory.
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", tempfile.mktemp(suffix=".db"))
    _set_selection(db, {"audit_sink": "memory"})
    from core.safeharness_audit import build_audit_sink

    sink = asyncio.run(_build(db))
    from safeharness.backends import InMemoryAuditSink

    assert isinstance(sink, InMemoryAuditSink)


async def _build(db):
    # build_audit_sink reads selection via its own default (None) session —
    # patch the registry's read to the shared engine for determinism.
    import core.safeharness_registry as reg

    original = reg.load_selection
    reg.load_selection = lambda db_=None: original(db)
    try:
        from core.safeharness_audit import build_audit_sink

        return build_audit_sink()
    finally:
        reg.load_selection = original


def test_audit_registry_auto_keeps_sqlite_resolution(db, monkeypatch, tmp_path):
    import tempfile

    audit_db = tmp_path / "audit.db"
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", str(audit_db))
    _set_selection(db, {"audit_sink": "auto"})
    from core.safeharness_audit import SQLiteAuditSink, build_audit_sink

    sink = asyncio.run(_build(db))
    assert isinstance(sink, SQLiteAuditSink)


def test_audit_registry_unknown_falls_back_to_auto(db, monkeypatch, tmp_path):
    import tempfile

    audit_db = tmp_path / "audit.db"
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", str(audit_db))
    _set_selection(db, {"audit_sink": "blackhole"})
    from core.safeharness_audit import SQLiteAuditSink, build_audit_sink

    sink = asyncio.run(_build(db))
    assert isinstance(sink, SQLiteAuditSink)


def test_content_gate_paranoid_never_touches_model_resolution(db, monkeypatch):
    from core.evolution import content_gate

    # The site reads the registry module's function at call time — patch there.
    monkeypatch.setattr(
        registry, "selected_judge_implementation", lambda db_=None: "paranoid"
    )

    def _boom():
        raise AssertionError("model resolution must not run in paranoid mode")

    monkeypatch.setattr(content_gate, "_resolve_judge", _boom)

    async def scenario():
        from core.evolution.content_gate import _inspect_semantic

        return await _inspect_semantic("良性内容", context={"session_id": "s"})

    decision = asyncio.run(scenario())
    assert decision.allowed is False  # fail-closed: UNCERTAIN blocks
    assert decision.reason_code == "semantic_prompt_injection_uncertain"
    assert decision.judge_status == "UNCERTAIN"  # paranoid's standing answer
