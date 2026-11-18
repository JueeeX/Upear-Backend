"""Runner sandbox (P3.2 pilot + P3.3 extension): policy dispatch, subprocess
isolation, timeout, per-framework worker dispatch.

Pins: default-off bit-identical dispatch, per-framework policy table
invariants (all three builtin frameworks subprocess), inprocess wall-clock
timeout (unknown-framework fallback), subprocess happy path (api_key via
stdin only), child env allowlist (parent secrets excluded), timeout kill on
the subprocess path, and fail-safe inprocess fallback for unknown frameworks.
Uses stub worker modules — no smolagents/agents/autogen or model dependency.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from core.external_agents import sandbox_runner
from core.external_agents import sandbox_worker
from core.external_agents import runners as runners_module
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
STUB_AGENTS_WORKER = "tests.sandbox_stub_worker_agents"


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


def _agent_request(retry_trigger: int = 0) -> dict:
    """A worker-protocol request as the parent writes it for openai-agents."""
    return {
        "framework": "openai-agents",
        "message": "hi",
        "history": [{"role": "user", "content": "h1"}],
        "long_term": "LT",
        "base_url": "http://relay/v1",
        "api_key": "sk-child-key",
        "model": "m",
        "supervision": {
            "max_steps": 3,
            "loop_trigger": 3,
            "max_output_chars": 8000,
            "max_total_chars": 24000,
            "retry_trigger": retry_trigger,
        },
        "run_id": "run-9",
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


def test_inprocess_timeout_degrades_to_halted():
    # The inprocess wall clock now only applies to unknown-framework fallback
    # (all builtin frameworks run subprocess workers when the gate is on).
    calls: list = []
    runner = _fake_runner(calls, delay=5.0)
    started = time.monotonic()
    result = asyncio.run(
        run_with_sandbox_policy(
            runner,
            "mystery-framework",
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


@pytest.mark.parametrize("framework", ["openai-agents", "autogen"])
def test_agent_framework_policy_dispatches_to_subprocess(monkeypatch, framework):
    """Sandbox on + builtin agent framework → the declared subprocess worker
    runs (framework field reaches the child), the in-process runner never does,
    and the isolation contract (stdin key, env allowlist) still holds."""
    monkeypatch.setenv("SAFEHARNESS_PROBE_KEY", "sk-parent-only-secret")
    monkeypatch.setattr(sandbox_runner, "WORKER_MODULE", STUB_AGENTS_WORKER)
    calls: list = []
    runner = _fake_runner(calls)
    result = asyncio.run(
        run_with_sandbox_policy(
            runner, framework, "ping", _config(), environ={ENV_RUNNER_SANDBOX: "1"}
        )
    )
    assert calls == []
    assert f"FW:{framework}" in result.reply
    assert "MSG:ping" in result.reply
    assert "KEY:sk-model-key-sentinel" in result.reply
    assert "sk-parent-only-secret" not in result.reply
    env_names = result.reply.split("ENV:")[1].split(";KEY:")[0].split(",")
    assert "SAFEHARNESS_PROBE_KEY" not in env_names
    assert result.framework == framework


@pytest.mark.parametrize(
    ("framework", "worker"), [("smolagents", STUB_WORKER), ("openai-agents", STUB_AGENTS_WORKER)]
)
def test_subprocess_timeout_kills_child(framework, worker, monkeypatch):
    if worker == STUB_AGENTS_WORKER:
        monkeypatch.setattr(sandbox_runner, "WORKER_MODULE", worker)
    started = time.monotonic()
    result = asyncio.run(
        run_subprocess_worker(
            "sleep:30",
            _config(),
            framework=framework,
            timeout_seconds=1.0,
            worker_module=worker,
        )
    )
    elapsed = time.monotonic() - started
    assert result.halted is True
    assert result.interventions[0]["trigger"] == "sandbox_timeout"
    assert elapsed < 15.0


def test_policy_table_covers_every_runner():
    assert set(RUNNERS) <= set(RUNNER_SANDBOX_POLICIES)
    for name, policy in RUNNER_SANDBOX_POLICIES.items():
        assert policy.mode in ("inprocess", "subprocess")
        assert policy.timeout_seconds > 0
    # Every builtin framework is subprocess-isolated when the gate is on.
    for name in RUNNERS:
        assert RUNNER_SANDBOX_POLICIES[name].mode == "subprocess"
        assert RUNNER_SANDBOX_POLICIES[name].env_passthrough


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


# ── Real worker dispatch (in-process unit tests, no framework libraries) ────


def _patch_gateway_bundle(monkeypatch, bundle):
    async def _factory(**kwargs):
        return bundle

    monkeypatch.setattr(
        "core.external_agents.gateway_tools.build_gateway_tool_bundle", _factory
    )


def test_worker_agent_dispatch_runs_runner_and_maps_payload(monkeypatch):
    seen: dict = {}

    async def _fake_runner(message: str, config: dict) -> RunnerResult:
        seen["message"] = message
        seen["config"] = config
        return RunnerResult(
            reply="done", framework="openai-agents", model="m", duration_ms=7, steps=2
        )

    monkeypatch.setattr(runners_module, "RUNNERS", {"openai-agents": _fake_runner})
    _patch_gateway_bundle(monkeypatch, None)
    payload = asyncio.run(sandbox_worker._run(_agent_request(retry_trigger=2)))
    assert payload["ok"] is True
    assert payload["reply"] == "done"
    assert payload["steps"] == 2
    assert payload["halted"] is False
    assert payload["interventions"] == []
    assert seen["message"] == "hi"
    config = seen["config"]
    assert config["api_key"] == "sk-child-key"
    assert config["base_url"] == "http://relay/v1"
    assert config["history"] == [{"role": "user", "content": "h1"}]
    assert config["long_term"] == "LT"
    assert config["run_id"] == "run-9"
    # Supervision rebuilt from the wire dict with the same thresholds.
    assert config["supervision"].config.max_steps == 3
    assert config["supervision"].config.retry_trigger == 2


def test_worker_agent_dispatch_wires_retry_sink_only_when_enabled(monkeypatch):
    configs: list[dict] = []

    async def _fake_runner(message: str, config: dict) -> RunnerResult:
        configs.append(config)
        return RunnerResult(reply="done", framework="openai-agents", model="m", duration_ms=1)

    def _run_case(retry_trigger: int, bundle: dict) -> dict:
        configs.clear()
        async def _fake(**_kwargs):
            return dict(bundle)

        monkeypatch.setattr(runners_module, "RUNNERS", {"openai-agents": _fake_runner})
        monkeypatch.setattr(
            "core.external_agents.gateway_tools.build_gateway_tool_bundle", _fake
        )
        payload = asyncio.run(sandbox_worker._run(_agent_request(retry_trigger=retry_trigger)))
        return {"payload": payload, "config": configs[-1]}

    wired = _run_case(2, {"ready": True})
    assert wired["payload"]["ok"] is True
    gateway_tool = wired["config"]["gateway_tool"]
    assert gateway_tool["on_denial"] == wired["config"]["supervision"].record_tool_denial

    plain = _run_case(0, {"ready": True})
    assert plain["payload"]["ok"] is True
    assert "on_denial" not in plain["config"]["gateway_tool"]


def test_worker_agent_dispatch_maps_runner_failure_to_failed_payload(monkeypatch):
    async def _boom(message: str, config: dict) -> RunnerResult:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(runners_module, "RUNNERS", {"autogen": _boom})
    _patch_gateway_bundle(monkeypatch, None)
    request = _agent_request()
    request["framework"] = "autogen"
    payload = asyncio.run(sandbox_worker._run(request))
    assert payload["ok"] is True
    assert payload["failed"] is True
    assert payload["error_type"] == "RuntimeError"
    assert "model exploded" in payload["error"]


def test_worker_agent_dispatch_unknown_runner_answers_protocol_error(monkeypatch):
    monkeypatch.setattr(runners_module, "RUNNERS", {})
    _patch_gateway_bundle(monkeypatch, None)
    payload = asyncio.run(sandbox_worker._run(_agent_request()))
    assert payload["ok"] is False
    assert payload["error_type"] == "ProtocolError"
    assert "openai-agents" in payload["error"]
