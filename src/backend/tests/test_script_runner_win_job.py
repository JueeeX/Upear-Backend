"""Windows Job Object caps for the script_runner sidecar (ticket #11).

The sidecar deploys standalone (no ``core.*`` imports), so its job helpers
live in ``services/script_runner_service/_win_job.py``, mirroring
``core.external_agents.sandbox_runner``. Pins: env parsing, child job
membership through a real ``_execute_subprocess`` round-trip, memory cap
enforcement (alloc breach), and active-process cap enforcement (grandchild
spawn denial / allowance).
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from services.script_runner_service import server
from services.script_runner_service._win_job import (
    ENV_JOB_MAX_PROCESSES,
    ENV_JOB_MEMORY_MB,
    job_limits_from_environ,
)


def test_job_limits_env_parsing():
    defaults = job_limits_from_environ({})
    assert defaults["memory_bytes"] == 2048 * 1024 * 1024
    assert defaults["max_processes"] == 64
    assert job_limits_from_environ({ENV_JOB_MEMORY_MB: "512"})["memory_bytes"] == 512 * 1024 * 1024
    assert job_limits_from_environ({ENV_JOB_MAX_PROCESSES: "16"})["max_processes"] == 16
    # 0 disables an individual cap; invalid falls back to the default; <1 clamps.
    assert job_limits_from_environ({ENV_JOB_MEMORY_MB: "0"})["memory_bytes"] is None
    assert job_limits_from_environ({ENV_JOB_MAX_PROCESSES: "bogus"})["max_processes"] == 64
    assert job_limits_from_environ({ENV_JOB_MAX_PROCESSES: "-3"})["max_processes"] == 1


_IN_JOB_PROBE = (
    "import ctypes, sys\n"
    "from ctypes import wintypes\n"
    "k = ctypes.WinDLL('kernel32')\n"
    "k.GetCurrentProcess.restype = wintypes.HANDLE\n"
    "k.IsProcessInJob.argtypes = (wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL))\n"
    "inj = wintypes.BOOL(0)\n"
    "ok = k.IsProcessInJob(k.GetCurrentProcess(), None, ctypes.byref(inj))\n"
    "print('IN_JOB:%d' % inj.value if ok else 'QUERY_FAILED')\n"
)

_SPAWN_PROBE = (
    "import subprocess, sys\n"
    "ok = 0\n"
    "for _ in range(2):\n"
    "    try:\n"
    "        p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.3)'])\n"
    "        p.wait(30)\n"
    "        ok += 1\n"
    "    except Exception:\n"
    "        pass\n"
    "print('SPAWNED', ok)\n"
)


def _run(code: str, tmp_path, timeout: int = 60) -> dict:
    return asyncio.run(
        server._execute_subprocess([sys.executable, "-c", code], "", timeout, str(tmp_path))
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object caps")
def test_child_is_in_job_with_default_caps(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_JOB_MEMORY_MB, raising=False)
    monkeypatch.delenv(ENV_JOB_MAX_PROCESSES, raising=False)
    result = _run(_IN_JOB_PROBE, tmp_path)
    assert result["exit_code"] == 0
    assert "IN_JOB:1" in result["stdout"]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object caps")
def test_memory_cap_blocks_alloc(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_JOB_MEMORY_MB, "8")
    code = (
        "blob = bytearray(32 * 1024 * 1024)\n"
        "for i in range(0, len(blob), 4096):\n"
        "    blob[i] = 1\n"
        "print('ALLOC_OK')\n"
    )
    result = _run(code, tmp_path)
    # The 8 MB cap must deny the 32 MB commit; the child never prints ALLOC_OK.
    assert "ALLOC_OK" not in result["stdout"]
    assert result["exit_code"] != 0


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object caps")
def test_process_cap_blocks_grandchildren(monkeypatch, tmp_path):
    # child(1) + two grandchildren = 3 live processes > ActiveProcessLimit 2:
    # CreateProcess must be denied for both attempts.
    monkeypatch.setenv(ENV_JOB_MAX_PROCESSES, "2")
    result = _run(_SPAWN_PROBE, tmp_path)
    assert "SPAWNED 0" in result["stdout"]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object caps")
def test_roomier_limit_allows_grandchildren(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_JOB_MAX_PROCESSES, "8")
    result = _run(_SPAWN_PROBE, tmp_path)
    assert "SPAWNED 2" in result["stdout"]
