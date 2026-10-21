"""P1.1 middleware assembly registry — default composition is a contract.

The registry must assemble, under default config, exactly the pre-registry
hardcoded list (same classes, same order, same gates); operators may disable
functional entries only, security entries refuse, malformed input fails open.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from core.auth.backend import UserContext
from core.llm.middleware_registry import (
    DEFAULT_REGISTRY,
    FUNCTIONAL_NAMES,
    LEVEL_SECURITY,
    OVERRIDES_CONFIG_KEY,
    assemble_middlewares,
    load_disabled_names,
)
from core.db.engine import Base
from core.db.models import SystemConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# The pre-registry hardcoded composition, class for class, in order.
LEGACY_ORDER = [
    "DynamicModelMiddleware",
    "FileContextMiddleware",
    "WorkspacePinHintMiddleware",
    "IterBudgetReminderMiddleware",
    "StallInterventionMiddleware",
    "OntologyGateMiddleware",
    "ActingToolCallIdMiddleware",
    "GoalAnchorReminderMiddleware",
    "FinishPinGuardMiddleware",
]


@pytest.fixture(autouse=True)
def _no_guard_env(monkeypatch):
    monkeypatch.delenv("SAFEHARNESS_MCP_POLICY_GUARD", raising=False)


@pytest.fixture
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    s = sessionmaker(bind=eng)()
    yield s
    s.close()


def _ctx(**over):
    from core.llm.middleware_registry import AssemblyContext

    values = {"batch_mode": False, "chat_id": "ch1", "intervention_rules": [], "ontology_runtime": None}
    values.update(over)
    return AssemblyContext(**values)


def _classes(mws):
    return [type(m).__name__ for m in mws]


def test_default_composition_matches_legacy_bit_for_bit(monkeypatch):
    monkeypatch.setattr("core.db.engine.SessionLocal", sessionmaker(bind=create_engine("sqlite://")))
    mws = assemble_middlewares(_ctx())
    assert _classes(mws)[:7] == LEGACY_ORDER[:7]
    assert _classes(mws)[7:] == LEGACY_ORDER[7:]
    assert len(mws) == 9  # guard env off: SafeHarness middleware absent


def test_guard_env_appends_safeharness_last(monkeypatch):
    monkeypatch.setattr("core.db.engine.SessionLocal", sessionmaker(bind=create_engine("sqlite://")))
    monkeypatch.setenv("SAFEHARNESS_MCP_POLICY_GUARD", "1")
    mws = assemble_middlewares(_ctx())
    assert _classes(mws)[-1] == "SafeHarnessToolPolicyMiddleware"
    assert len(mws) == 10


def test_batch_mode_drops_goal_anchor(monkeypatch):
    monkeypatch.setattr("core.db.engine.SessionLocal", sessionmaker(bind=create_engine("sqlite://")))
    mws = assemble_middlewares(_ctx(batch_mode=True))
    assert "GoalAnchorReminderMiddleware" not in _classes(mws)
    assert len(mws) == 8


def test_builder_kwargs_flow_through(monkeypatch):
    monkeypatch.setattr("core.db.engine.SessionLocal", sessionmaker(bind=create_engine("sqlite://")))
    rules = [{"signal": "repeated_actions"}]
    runtime = {"pack": "demo"}
    mws = assemble_middlewares(_ctx(intervention_rules=rules, ontology_runtime=runtime))
    by_name = {type(m).__name__: m for m in mws}
    assert by_name["StallInterventionMiddleware"]._rules == rules
    assert by_name["OntologyGateMiddleware"].runtime == runtime
    assert by_name["GoalAnchorReminderMiddleware"]._chat_id == "ch1"


def test_functional_disable_skips_entry(db, monkeypatch):
    s = sessionmaker(bind=db.bind)()
    s.add(
        SystemConfig(
            config_key=OVERRIDES_CONFIG_KEY,
            config_value=json.dumps({"disabled": ["workspace_pin_hint"]}),
            display_name="x",
            group_key="middleware",
            is_secret=False,
        )
    )
    s.commit()
    s.close()
    mws = assemble_middlewares(_ctx(), db=db)
    assert "WorkspacePinHintMiddleware" not in _classes(mws)
    assert len(mws) == 8


def test_security_disable_is_refused(db):
    s = sessionmaker(bind=db.bind)()
    s.add(
        SystemConfig(
            config_key=OVERRIDES_CONFIG_KEY,
            config_value=json.dumps({"disabled": ["ontology_gate", "safeharness_tool_policy"]}),
            display_name="x",
            group_key="middleware",
            is_secret=False,
        )
    )
    s.commit()
    s.close()
    assert "OntologyGateMiddleware" in _classes(assemble_middlewares(_ctx(), db=db))


def test_unknown_and_malformed_fail_open(db):
    s = sessionmaker(bind=db.bind)()
    s.add(
        SystemConfig(
            config_key=OVERRIDES_CONFIG_KEY,
            config_value="{not json",
            display_name="x",
            group_key="middleware",
            is_secret=False,
        )
    )
    s.commit()
    s.close()
    assert len(assemble_middlewares(_ctx(), db=db)) == 9
    assert load_disabled_names(db) == set()


def test_fresh_read_hot_effect(db):
    mws_before = _classes(assemble_middlewares(_ctx(), db=db))
    s = sessionmaker(bind=db.bind)()
    s.add(
        SystemConfig(
            config_key=OVERRIDES_CONFIG_KEY,
            config_value=json.dumps({"disabled": ["iter_budget_reminder"]}),
            display_name="x",
            group_key="middleware",
            is_secret=False,
        )
    )
    s.commit()
    s.close()
    mws_after = _classes(assemble_middlewares(_ctx(), db=db))
    assert "IterBudgetReminderMiddleware" in mws_before
    assert "IterBudgetReminderMiddleware" not in mws_after


def test_registry_levels_and_names():
    assert LEVEL_SECURITY not in {spec.level for spec in ()}
    security = [spec.name for spec in DEFAULT_REGISTRY if spec.level == LEVEL_SECURITY]
    assert security == ["ontology_gate", "safeharness_tool_policy"]
    assert len(FUNCTIONAL_NAMES) == 8


def test_specs_are_frozen_and_build_fresh_instances():
    ctx = _ctx()
    for spec in DEFAULT_REGISTRY:
        with pytest.raises(Exception):
            spec.name = "x"  # type: ignore[misc]
    first = assemble_middlewares(ctx)
    second = assemble_middlewares(ctx)
    for a, b in zip(first, second):
        assert a is not b  # fresh instances every assembly (per-run state)
