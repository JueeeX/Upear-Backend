"""Windows Job Object caps for script_runner child executions (ticket #11).

Self-contained on purpose: the script_runner service deploys standalone
(stdlib + fastapi + pydantic only, no ``core.*`` on its PYTHONPATH), so this
mirrors ``core.external_agents.sandbox_runner``'s job helpers instead of
importing them. Keep the two in sync — same env names, same defaults, same
fail-open semantics — so operators have one knob story:

- ``SAFEHARNESS_RUNNER_JOB_MEMORY_MB``   per-process memory cap (default 2048, 0 = off)
- ``SAFEHARNESS_RUNNER_JOB_MAX_PROCESSES`` live-process cap (default 64, 0 = off)

The child (and every descendant it spawns) is assigned to a Job Object with
the configured caps plus KILL_ON_JOB_CLOSE, so the tree dies with the service
even if the wall-clock kill never fires. Assignment is best-effort: if it
fails, the execution proceeds behind the existing process-group kill and
``SAFE_ENV`` allowlist and the failure is logged — the caps are defence in
depth, not the only fence. POSIX keeps the documented shared-UID boundary
(RLIMIT_NPROC only where the deployment profile says it is safe).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENV_JOB_MEMORY_MB = "SAFEHARNESS_RUNNER_JOB_MEMORY_MB"
ENV_JOB_MAX_PROCESSES = "SAFEHARNESS_RUNNER_JOB_MAX_PROCESSES"
DEFAULT_JOB_MEMORY_MB = 2048
DEFAULT_JOB_MAX_PROCESSES = 64

_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_INFO_CLASS_EXTENDED_LIMIT = 9


def job_limits_from_environ(environ: Optional[dict] = None) -> dict:
    """Parsed caps: ``{"memory_bytes": int|None, "max_processes": int|None}``.

    ``None`` = cap disabled. Invalid values fall back to the defaults; an
    explicit ``0`` disables that individual cap. Pure function.
    """
    values = os.environ if environ is None else environ

    def _parse(name: str, default: int) -> Optional[int]:
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

    memory_mb = _parse(ENV_JOB_MEMORY_MB, DEFAULT_JOB_MEMORY_MB)
    max_processes = _parse(ENV_JOB_MAX_PROCESSES, DEFAULT_JOB_MAX_PROCESSES)
    return {
        "memory_bytes": None if memory_mb is None else memory_mb * 1024 * 1024,
        "max_processes": max_processes,
    }


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

    return (kernel32, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, ctypes, wintypes.HANDLE)


def assign_windows_job(process: Any, limits: dict) -> Any:
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
        # asyncio transports expose the Popen either as ``_proc`` (this
        # Python) or via ``get_proc()`` (other versions) — cover both.
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
            "[script-runner] job-object assignment skipped (%s); process-group "
            "kill and SAFE_ENV remain the enforced fences",
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
