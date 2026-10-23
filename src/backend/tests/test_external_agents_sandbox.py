"""Runner sandbox pilot (P3.2): policy dispatch, subprocess isolation, timeout.

Pins: default-off bit-identical dispatch, per-framework policy table
invariants, inprocess wall-clock timeout, subprocess happy path (api_key via
stdin only), child env allowlist (parent secrets excluded), timeout kill on
the subprocess path, and fail-safe inprocess fallback for unknown frameworks.
Uses a stub worker module — no smolagents or model dependency.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from core.external_agents.runners import (
    RUNNERS,
    RUNNER_SANDBOX_POLICIES,
    RunnerResult,
    RunnerSandboxPolicy,
    run_with_sandbox_policy,
)
from core.external_agents.sandbox_runner import (
    ENV_RUNNER_SANDBOX,
    ENV_RUNNER_TIMEOUT,
    run_subprocess_worker,
    runner_timeout_seconds,
    sandbox_enabled,
)
from core.external_agents.supervisor import DEFAULT_SUPERVISION, SupervisionState

STUB_WORKER = "tests.sandbox_stub_worker"


def _result(framework: str = "fake") -> RunnerResult:
    return RunnerResult(reply="ok", framework=framework, model="m", duration_ms=1)


def _fake_runner(calls: list[tuple[str, dict]], reply: str = "ok", delay: float = 0.0):
    async def _run(message: str, config: dict) -> RunnerResult:
        if delay:
            await asyncio.sleep(delay)
        calls.append((message, config))
        return RunnerResult(reply=reply, framework="fake", model="m", duration_ms=1)

    return _run


def _config() -> dict:
    return {
        "api_key": "sk-model-key-sentinel",
        "model": "test-model",
        "supervision": SupervisionState(config=DEFAULT_SUPERVISION),
        "run_id": "run-1",
    }


def test_sandbox_disabled_is_bare_passthrough(monkeypatch):
    monkeypatch.delenv(ENV_RUNNER_SANDBOX, raising=False)
    assert sandbox_enabled({}) is False
    calls: list = []
    runner = _fake_runner(calls)
    result = asyncio.run(
        run_with_sandbox_policy(runner, "smolagents", "hi", _config(), environ={})
    )
    assert result.reply == "ok"
    assert len(calls) == 1


def test_sandbox_enabled_inprocess_mode_invokes_runner(monkeypatch):
    calls: list = []
    runner = _fake_runner(calls)
    result = asyncio.run(
        run_with_sandbox_policy(
            runner, "autogen", "hi", _config(), environ={ENV_RUNNER_SANDBOX: "1"}
        )
    )
    assert result.reply == "ok"
    assert len(calls) == 1


def test_inprocess_timeout_degrades_to_halted():
    calls: list = []
    runner = _fake_runner(calls, delay=5.0)
    started = time.monotonic()
    result = asyncio.run(
        run_with_sandbox_policy(
            runner,
            "autogen",
            "hi",
            _config(),
            environ={ENV_RUNNER_SANDBOX: "1", ENV_RUNNER_TIMEOUT: "0.5"},
        )
    )
    elapsed = time.monotonic() - started
    assert result.halted is True and result.reply == ""
    assert result.interventions[0]["trigger"] == "sandbox_timeout"
    assert elapsed < 4.0


def test_subprocess_happy_path_key_via_stdin(monkeypatch):
    monkeypatch.setenv("SAFEHARNESS_PROBE_KEY", "sk-parent-only-secret")
    result = asyncio.run(
        run_subprocess_worker(
            "ping",
            _config(),
            framework="smolagents",
            timeout_seconds=30,
            worker_module=STUB_WORKER,
        )
    )
    # The model key travelled through stdin (memory), not the child env.
    assert "KEY:sk-model-key-sentinel" in result.reply
    assert "sk-parent-only-secret" not in result.reply
    env_names = result.reply.split("ENV:")[1].split(";KEY:")[0].split(",")
    assert "PATH" in env_names
    assert "SAFEHARNESS_PROBE_KEY" not in env_names
    assert result.framework == "smolagents"


def test_subprocess_timeout_kills_child():
    started = time.monotonic()
    result = asyncio.run(
        run_subprocess_worker(
            "sleep:30",
            _config(),
            framework="smolagents",
            timeout_seconds=1.0,
            worker_module=STUB_WORKER,
        )
    )
    elapsed = time.monotonic() - started
    assert result.halted is True
    assert result.interventions[0]["trigger"] == "sandbox_timeout"
    assert elapsed < 15.0


def test_unknown_framework_falls_back_to_inprocess(monkeypatch):
    calls: list = []
    runner = _fake_runner(calls)
    result = asyncio.run(
        run_with_sandbox_policy(
            runner, "mystery-framework", "hi", _config(), environ={ENV_RUNNER_SANDBOX: "1"}
        )
    )
    assert result.reply == "ok"
    assert len(calls) == 1


def test_policy_table_covers_every_runner():
    assert set(RUNNERS) <= set(RUNNER_SANDBOX_POLICIES)
    for name, policy in RUNNER_SANDBOX_POLICIES.items():
        assert policy.mode in ("inprocess", "subprocess")
        assert policy.timeout_seconds > 0
    assert RUNNER_SANDBOX_POLICIES["smolagents"].mode == "subprocess"


def test_timeout_env_parsing_bounds():
    assert runner_timeout_seconds({}) == 120.0
    assert runner_timeout_seconds({ENV_RUNNER_TIMEOUT: "30"}) == 30.0
    assert runner_timeout_seconds({ENV_RUNNER_TIMEOUT: "bogus"}) == 120.0
    assert runner_timeout_seconds({ENV_RUNNER_TIMEOUT: "0"}) == 120.0
    assert runner_timeout_seconds({ENV_RUNNER_TIMEOUT: "9999"}) == 600.0


def test_subprocess_worker_crash_raises(tmp_path):
    broken = tmp_path / "broken_worker.py"
    broken.write_text("raise SystemExit(3)\n", encoding="utf-8")
    with pytest.raises(Exception):
        asyncio.run(
            run_subprocess_worker(
                "hi",
                _config(),
                framework="smolagents",
                timeout_seconds=30,
                worker_module="sumc_broken_worker_never_importable",
            )
        )
