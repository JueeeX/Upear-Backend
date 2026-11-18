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
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from core.external_agents.runners import RunnerResult

logger = logging.getLogger(__name__)

ENV_RUNNER_SANDBOX = "SAFEHARNESS_RUNNER_SANDBOX"
ENV_RUNNER_TIMEOUT = "SAFEHARNESS_RUNNER_TIMEOUT_SECONDS"
# Windows Job Object caps (ticket #11): memory per child process and total
# live processes inside the job. 0 disables an individual cap; invalid values
# fall back to the defaults. POSIX keeps the documented shared-UID boundary —
# no RLIMIT, the wall-clock kill remains the only hard stop there.
ENV_RUNNER_JOB_MEMORY_MB = "SAFEHARNESS_RUNNER_JOB_MEMORY_MB"
ENV_RUNNER_JOB_MAX_PROCESSES = "SAFEHARNESS_RUNNER_JOB_MAX_PROCESSES"

DEFAULT_JOB_MEMORY_MB = 2048
DEFAULT_JOB_MAX_PROCESSES = 64

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


# ── Windows Job Object caps (ticket #11) ─────────────────────────────────────
# The sandbox child (and every descendant it spawns) is assigned to a Job
# Object with a per-process memory cap and an active-process cap, plus
# KILL_ON_JOB_CLOSE so the tree dies with the parent even if the wall-clock
# kill never fires. Best-effort: if the assignment itself fails, the run
# proceeds with the existing env allowlist + wall-clock kill and the failure
# is logged — the caps are defence in depth, not the only fence.

_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_INFO_CLASS_EXTENDED_LIMIT = 9


def _job_object_apis():
    """ctypes bindings; called only on Windows."""
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPWSTR, wintypes.LPWSTR)
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    )
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    return (
        kernel32,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes,
        wintypes.HANDLE,
    )


def job_limits_from_environ(environ: dict[str, str] | None = None) -> dict[str, int | None]:
    """Parsed job caps: ``{"memory_bytes": int|None, "max_processes": int|None}``.

    ``None`` = cap disabled. Invalid values fall back to the defaults; an
    explicit ``0`` disables that individual cap. Pure function — pinned by
    tests on every platform.
    """
    values = os.environ if environ is None else environ

    def _parse(name: str, default: int) -> int | None:
        raw = (values.get(name) or "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        if value == 0:
            return None
        return max(value, 1)

    memory_mb = _parse(ENV_RUNNER_JOB_MEMORY_MB, DEFAULT_JOB_MEMORY_MB)
    max_processes = _parse(ENV_RUNNER_JOB_MAX_PROCESSES, DEFAULT_JOB_MAX_PROCESSES)
    return {
        "memory_bytes": None if memory_mb is None else memory_mb * 1024 * 1024,
        "max_processes": max_processes,
    }


def assign_windows_job(process: Any, limits: dict[str, int | None]) -> Any:
    """Put the child (and its future descendants) under the job caps.

    Returns the job handle the caller must keep open until the child is done
    (KILL_ON_JOB_CLOSE fires on the final CloseHandle), or ``None`` when not
    on Windows, when no cap is enabled, or when the assignment fails.
    """
    if os.name != "nt":
        return None
    memory_bytes = limits.get("memory_bytes")
    max_processes = limits.get("max_processes")
    if not memory_bytes and not max_processes:
        return None
    try:
        kernel32, extended_info, ctypes, handle_type = _job_object_apis()

        info = extended_info()
        limit_flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_bytes:
            limit_flags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = memory_bytes
        if max_processes:
            limit_flags |= _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            info.BasicLimitInformation.ActiveProcessLimit = max_processes
        info.BasicLimitInformation.LimitFlags = limit_flags

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError("CreateJobObjectW failed")
        if not kernel32.SetInformationJobObject(
            job, _JOB_OBJECT_INFO_CLASS_EXTENDED_LIMIT, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise OSError("SetInformationJobObject failed")
        transport = getattr(process, "_transport", None)
        popen = getattr(transport, "_proc", None)
        if popen is None and hasattr(transport, "get_proc"):
            popen = transport.get_proc()
        handle = int(getattr(popen, "_handle", 0) or 0)
        if not handle:
            raise OSError("child process handle unavailable")
        if not kernel32.AssignProcessToJobObject(job, handle_type(handle)):
            raise OSError("AssignProcessToJobObject failed")
        return job
    except Exception as exc:  # noqa: BLE001 — caps are defence in depth
        logger.warning(
            "[sandbox] job-object assignment skipped (%s); wall-clock kill and "
            "env allowlist remain the enforced fences",
            type(exc).__name__,
        )
        return None


def close_job_handle(job: Any) -> None:
    """Release the job (fires KILL_ON_JOB_CLOSE if the child somehow lives on)."""
    if not job or os.name != "nt":
        return
    try:
        kernel32, *_ = _job_object_apis()
        kernel32.CloseHandle(job)
    except Exception:  # noqa: BLE001 — cleanup noise must not mask results
        pass


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
    worker_module: str | None = None,
) -> RunnerResult:
    """Run the framework worker in a child process with a hard wall clock.

    The result JSON travels via a temp file, not stdout: framework libraries
    print to stdout even at verbosity 0, and the protocol channel must stay
    exclusive (same temp-file-stdio discipline as the script-runner sidecar).
    The file holds the reply only — the api_key goes in via stdin and is never
    written anywhere.
    """
    # Resolved at call time (not as a default argument) so tests can retarget
    # the worker with a stub via monkeypatching the module constant.
    worker_module = worker_module or WORKER_MODULE
    request = {
        "message": message,
        # Which runner the child should execute (additive P3.3: the smolagents
        # worker ignores it; the dispatching worker needs it). Absent ⇒ legacy
        # smolagents path, so pre-existing workers keep working unchanged.
        "framework": str(framework or ""),
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
    # Ticket #11 (Windows): memory / process-count caps via a Job Object.
    # Best-effort assignment; the handle stays open for the child's lifetime.
    job = assign_windows_job(proc, job_limits_from_environ())
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
        close_job_handle(job)
        try:
            os.unlink(result_path)
        except OSError:
            pass
