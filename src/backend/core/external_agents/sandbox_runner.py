"""Parent-side subprocess sandbox for external-agent runners (P3.2).

Spawn → stdin JSON → ``asyncio.wait_for`` timeout → process-group kill →
stdout JSON. The kill discipline mirrors the script-runner sidecar exactly
(POSIX ``start_new_session`` + ``os.killpg``; Windows ``CREATE_NEW_PROCESS_
GROUP`` + ``taskkill /T /F`` with a ``proc.kill()`` fallback), so a hung or
runaway framework loop can never outlive its wall-clock budget.

The child environment is an explicit allowlist plus the runner policy's
passthrough names: the backend's full environment (DB URLs, platform keys)
never reaches the code-executing child. The model api_key travels via stdin
JSON — memory only, never env, never disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from core.external_agents.runners import RunnerResult

ENV_RUNNER_SANDBOX = "SAFEHARNESS_RUNNER_SANDBOX"
ENV_RUNNER_TIMEOUT = "SAFEHARNESS_RUNNER_TIMEOUT_SECONDS"

WORKER_MODULE = "core.external_agents.sandbox_worker"

# Allowlisted names copied into the child environment. PYTHONPATH is required
# so ``python -m core.external_agents.sandbox_worker`` resolves; everything
# else is child-process hygiene, not secrets.
_ENV_ALLOWLIST = (
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUTF8",
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "WINDIR",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERDOMAIN",
    "COMPUTERNAME",
)

# Gateway tool-surface envs the worker may need to rebuild the bundle; a
# runner policy lists exactly which of these its framework requires.
GATEWAY_ENV_PASSTHROUGH = (
    "SAFEHARNESS_TOOL_GATEWAY_URL",
    "SAFEHARNESS_TOOL_GATEWAY_API_KEY",
    "SAFEHARNESS_ENFORCEMENT_RECEIPT_MODE",
    "SAFEHARNESS_ENFORCEMENT_RECEIPT_SIGNING_KEY",
    "SAFEHARNESS_DEMO_TENANT_ID",
    "SAFEHARNESS_COMMUNICATION_AGENT_ID",
)


def _env_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes")


def sandbox_enabled(environ: dict[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return _env_flag(values.get(ENV_RUNNER_SANDBOX))


def runner_timeout_seconds(
    environ: dict[str, str] | None = None, default: float = 120.0
) -> float:
    values = os.environ if environ is None else environ
    try:
        value = float((values.get(ENV_RUNNER_TIMEOUT) or "").strip() or default)
    except ValueError:
        return default
    if value <= 0:
        return default
    return min(value, 600.0)


def _child_environment(env_passthrough: tuple[str, ...]) -> dict[str, str]:
    env: dict[str, str] = {}
    for name in _ENV_ALLOWLIST:
        value = os.environ.get(name)
        if value:
            env[name] = value
    for name in env_passthrough:
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def _spawn_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    if os.name == "nt":
        try:
            kill = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(proc.pid), "/T", "/F"
            )
            await kill.wait()
            return
        except Exception:  # noqa: BLE001 — fall through to proc.kill
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


async def run_subprocess_worker(
    message: str,
    config: dict[str, Any],
    *,
    framework: str,
    env_passthrough: tuple[str, ...] = (),
    timeout_seconds: float = 120.0,
    worker_module: str = WORKER_MODULE,
) -> RunnerResult:
    """Run the framework worker in a child process with a hard wall clock.

    The result JSON travels via a temp file, not stdout: framework libraries
    print to stdout even at verbosity 0, and the protocol channel must stay
    exclusive (same temp-file-stdio discipline as the script-runner sidecar).
    The file holds the reply only — the api_key goes in via stdin and is never
    written anywhere.
    """
    request = {
        "message": message,
        "history": list(config.get("history") or []),
        "long_term": str(config.get("long_term") or ""),
        "base_url": str(config.get("base_url") or ""),
        "api_key": str(config.get("api_key") or ""),
        "model": str(config.get("model") or ""),
        "supervision": (config.get("supervision") and config["supervision"].config.to_dict()) or {},
        "run_id": str(config.get("run_id") or ""),
    }
    result_fd, result_path = tempfile.mkstemp(prefix="upear-sandbox-result-", suffix=".json")
    os.close(result_fd)
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        worker_module,
        result_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=_child_environment(env_passthrough),
        **_spawn_kwargs(),
    )
    timed_out = False
    try:
        await asyncio.wait_for(
            proc.communicate(input=json.dumps(request, ensure_ascii=False).encode("utf-8")),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        timed_out = True
        await _kill_process_group(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            pass
    duration_ms = int((time.monotonic() - started) * 1000)
    try:
        if timed_out:
            return RunnerResult(
                reply="",
                framework=framework,
                model=str(config.get("model") or ""),
                duration_ms=duration_ms,
                halted=True,
                interventions=[
                    {
                        "trigger": "sandbox_timeout",
                        "detail": f"运行超时 {timeout_seconds:.0f}s，子进程已被沙箱终止",
                        "step": 0,
                    }
                ],
            )
        try:
            payload = json.loads(Path(result_path).read_text(encoding="utf-8").strip() or "{}")
        except (OSError, ValueError, UnicodeDecodeError):
            payload = {"ok": False, "error_type": "ProtocolError", "error": "worker 未写出结果文件"}
        if not payload.get("ok"):
            raise RuntimeError(
                f"sandbox worker failed: {payload.get('error_type')}: {payload.get('error')}"
            )
        if payload.get("failed"):
            raise RuntimeError(
                f"{framework} run failed: {payload.get('error_type')}: {payload.get('error')}"
            )
        return RunnerResult(
            reply=str(payload.get("reply") or ""),
            framework=framework,
            model=str(config.get("model") or ""),
            duration_ms=duration_ms,
            steps=int(payload.get("steps") or 0),
            interventions=list(payload.get("interventions") or []),
            halted=bool(payload.get("halted")),
        )
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass
