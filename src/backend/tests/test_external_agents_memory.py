"""External-agent memory: configurable session window + mem0 long-term layer.

The memory config row degrades to safe defaults on absence or corruption; the
long-term layer is budgeted (character cap at the injection point), keyed by
the platform user id, and contributes nothing when mem0 is disabled — a broken
memory path must never fail an external-agent run.
"""

import json

from core.external_agents.memory_settings import (
    MemoryConfig,
    load_memory_config,
    parse_memory_config,
    save_memory_config,
)
from core.external_agents.registry import CATALOG
from core.external_agents.runners import RUNNERS, RunnerResult


import contextlib


@contextlib.contextmanager
def _mem0_enabled(flag: bool):
    """settings is frozen; flip the mem0 bit through the dataclass bypass."""
    import core.config.settings as settings_module

    mem = settings_module.settings.memory
    original = mem.enabled
    object.__setattr__(mem, "enabled", flag)
    try:
        yield
    finally:
        object.__setattr__(mem, "enabled", original)



def _db():
    from core.db.engine import Base
    from core.db.models import SystemConfig
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine, tables=[SystemConfig.__table__])
    return sessionmaker(bind=engine)()


def test_memory_config_defaults_and_clamping():
    defaults = parse_memory_config(None)
    assert defaults == MemoryConfig()
    # Values out of range clamp; bad types fall back.
    clamped = parse_memory_config(
        {"history_window": 999, "long_term_max_chars": -5, "long_term_limit": "x",
         "long_term_enabled": "yes"}
    )
    assert clamped.history_window == 50
    assert clamped.long_term_max_chars == 0
    assert clamped.long_term_limit == 5
    assert clamped.long_term_enabled is True  # non-bool falls back


def test_memory_config_persists_and_falls_back():
    db = _db()
    assert load_memory_config(db) == MemoryConfig()
    merged = save_memory_config(db, {"history_window": 4, "long_term_max_chars": 800})
    assert (merged.history_window, merged.long_term_max_chars) == (4, 800)
    loaded = load_memory_config(db)
    assert (loaded.history_window, loaded.long_term_max_chars) == (4, 800)
    # Partial update keeps other fields.
    merged2 = save_memory_config(db, {"long_term_enabled": False})
    assert merged2.history_window == 4
    assert merged2.long_term_enabled is False
    # Corrupt row → defaults.
    from core.db.models import SystemConfig

    row = db.get(SystemConfig, "external_agent_memory")
    row.config_value = "{broken"
    db.commit()
    assert load_memory_config(db) == MemoryConfig()


def _point_runner(agent_id: str, spy):
    object.__setattr__(CATALOG[agent_id], "runner", spy)


async def test_registry_window_and_long_term_flow(monkeypatch):
    """Window truncation, retrieval injection, budget cap, and write scheduling."""
    captured: dict = {}

    async def _spy_runner(message, config):
        captured["config"] = config
        return RunnerResult(reply="好的", framework="autogen", model="m", duration_ms=1)

    async def _fake_retrieve(user_id, query, limit=10, min_score=0.4, **_kw):
        captured["retrieve"] = {"user_id": user_id, "query": query, "limit": limit}
        return "用户偏好：简洁回答。" * 500  # far over any sane budget

    saved: list[tuple[str, str, str]] = []

    async def _fake_save(user_id, user_message, assistant_message):
        saved.append((user_id, user_message, assistant_message))

    monkeypatch.setattr(
        "core.external_agents.registry._model_provider_config",
        lambda agent_id: {"base_url": "https://relay.example/v1", "api_key": "k",
                          "model": "m", "provider_id": "p", "source": "main_agent"},
    )
    import core.external_agents.registry as registry

    monkeypatch.setattr("core.db.engine.SessionLocal", _db)
    monkeypatch.setattr("core.memory.service.retrieve_memories", _fake_retrieve)
    monkeypatch.setattr("core.memory.service.save_conversation", _fake_save)

    _point_runner("ext-autogen", _spy_runner)
    history = [{"role": "user", "content": f"turn {i}"} for i in range(30)]
    with _mem0_enabled(True):
        data = await registry.run_external_agent(
            "ext-autogen",
            "记一下：我叫小焦",
            context={"user_id": "user-a", "tenant_id": "t", "session_id": "s", "trace_id": "tr"},
            history=history,
        )
    config = captured["config"]
    # Window: default 12 turns, tail of the 30-turn history.
    assert len(config["history"]) == 12
    assert config["history"][-1]["content"] == "turn 29"
    # Long-term text was injected and capped at the default budget.
    assert "用户偏好" in config["long_term"]
    assert len(config["long_term"]) <= MemoryConfig().long_term_max_chars
    assert captured["retrieve"]["limit"] == MemoryConfig().long_term_limit
    evidence = data["evidence"]["memory"]
    assert evidence["window_turns"] == 12
    assert evidence["long_term_read"] == "retrieved"
    assert evidence["long_term_write"] == "scheduled"
    # Save is fire-and-forget; yield one loop tick to let the task run.
    import asyncio

    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert saved and saved[0][0] == "user-a" and "小焦" in saved[0][1] and saved[0][2] == "好的"


async def test_registry_long_term_disabled_or_platform_off(monkeypatch):
    captured: dict = {}

    async def _spy_runner(message, config):
        captured["config"] = config
        return RunnerResult(reply="ok", framework="autogen", model="m", duration_ms=1)

    monkeypatch.setattr(
        "core.external_agents.registry._model_provider_config",
        lambda agent_id: {"base_url": "https://relay.example/v1", "api_key": "k",
                          "model": "m", "provider_id": "p", "source": "main_agent"},
    )
    import core.external_agents.registry as registry

    monkeypatch.setattr("core.db.engine.SessionLocal", _db)

    _point_runner("ext-autogen", _spy_runner)
    with _mem0_enabled(False):
        data = await registry.run_external_agent(
            "ext-autogen", "hi",
            context={"user_id": "user-b", "tenant_id": "t", "session_id": "s", "trace_id": "tr"},
            history=[{"role": "user", "content": "1"}],
        )
    assert "long_term" not in captured["config"]
    assert data["evidence"]["memory"]["long_term_read"] == "platform_disabled"

    # Config-level off with platform on: config gate wins, status stays disabled.
    db = _db()
    save_memory_config(db, {"long_term_enabled": False})
    monkeypatch.setattr("core.db.engine.SessionLocal", lambda: db)
    with _mem0_enabled(True):
        data2 = await registry.run_external_agent(
            "ext-autogen", "hi",
            context={"user_id": "user-b", "tenant_id": "t", "session_id": "s", "trace_id": "tr"},
            history=[{"role": "user", "content": "1"}],
        )
    assert "long_term" not in captured["config"]
    assert data2["evidence"]["memory"]["long_term_read"] == "disabled"


async def test_registry_window_config_applies(monkeypatch):
    """A stored window size truncates the injected history."""
    captured: dict = {}

    async def _spy_runner(message, config):
        captured["config"] = config
        return RunnerResult(reply="ok", framework="autogen", model="m", duration_ms=1)

    monkeypatch.setattr(
        "core.external_agents.registry._model_provider_config",
        lambda agent_id: {"base_url": "https://relay.example/v1", "api_key": "k",
                          "model": "m", "provider_id": "p", "source": "main_agent"},
    )
    import core.external_agents.registry as registry

    db = _db()
    save_memory_config(db, {"history_window": 2})
    monkeypatch.setattr("core.db.engine.SessionLocal", lambda: db)
    _point_runner("ext-autogen", _spy_runner)
    await registry.run_external_agent(
        "ext-autogen", "hi",
        context={"user_id": "", "tenant_id": "t", "session_id": "s", "trace_id": "tr"},
        history=[{"role": "user", "content": str(i)} for i in range(10)],
    )
    assert len(captured["config"]["history"]) == 2
    assert captured["config"]["history"][-1]["content"] == "9"


def test_runner_long_term_formatters():
    from core.external_agents.runners import (
        _CONFIDENTIALITY_CLAUSE,
        _long_term_instructions,
        openai_agents_input,
        smolagents_task,
    )

    task = smolagents_task([{"role": "user", "content": "a"}], "m", "记忆段")
    assert task.startswith(_CONFIDENTIALITY_CLAUSE)
    assert "记忆段\n\nm\n\n" in task
    assert smolagents_task([], "m", "").startswith(_CONFIDENTIALITY_CLAUSE)
    assert "m" in smolagents_task([], "m", "")
    assert _long_term_instructions("") == ""
    assert "长期记忆" in _long_term_instructions("X")
    # Native input list is unaffected by long-term (instructions carry it).
    inputs = openai_agents_input([{"role": "user", "content": "a"}], "b")
    assert inputs[-1] == {"role": "user", "content": "b"}
    _ = json  # imported for symmetry with bridge tests
