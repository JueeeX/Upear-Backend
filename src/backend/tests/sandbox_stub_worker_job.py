"""Job-object probe stub for the sandbox caps tests (ticket #11).

Same stdio protocol as the other sandbox stubs, but the reply reports the
child's Windows job membership: ``IN_JOB:<0|1>`` — plus a ``spawn:<n>``
directive that tries to create n grandchild processes and reports
``SPAWN_OK`` or ``SPAWN_FAIL:<k>`` (an ActiveProcessLimit breach makes
CreateProcess fail). Non-Windows: membership reports 0.

Caps themselves are verified behaviourally by the tests: an ActiveProcessLimit
breach blocks the spawn directive; a ProcessMemoryLimit below the interpreter's
boot footprint kills the child before it can write a result file.
"""

from __future__ import annotations

import json
import subprocess
import sys


def _probe() -> str:
    if sys.platform != "win32":
        return "IN_JOB:0"
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    kernel32.IsProcessInJob.argtypes = (
        wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL),
    )
    in_job = wintypes.BOOL(0)
    ok = kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), None, ctypes.byref(in_job))
    # ok=False means the query itself failed (INVALID_HANDLE etc.) — report it
    # as unknown rather than pretending the process is outside every job.
    return f"IN_JOB:{int(in_job.value)}" if ok else "IN_JOB:?;QUERY_FAILED"


def _spawn_children(count: int) -> str:
    failures = 0
    for _ in range(max(0, count)):
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(0.3)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001 — any failure counts as blocked
            failures += 1
    return "SPAWN_OK" if failures == 0 else f"SPAWN_FAIL:{failures}"


def main() -> int:
    request = json.loads(sys.stdin.read())
    message = str(request.get("message") or "")
    reply = _probe()
    if message.startswith("spawn:"):
        try:
            reply += ";" + _spawn_children(int(message.split(":", 1)[1]))
        except ValueError:
            reply += ";SPAWN_FAIL:parse"
    elif message.startswith("alloc:"):
        # Commit (not just reserve) mb of memory by touching every page — a
        # ProcessMemoryLimit below mb makes this raise MemoryError, which is
        # exactly what the enforcement test wants to observe.
        mb = int(message.split(":", 1)[1])
        blob = bytearray(mb * 1024 * 1024)
        for offset in range(0, len(blob), 4096):
            blob[offset] = 1
        reply += f";ALLOC:{mb}"
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "ok": True,
                    "reply": reply,
                    "steps": 1,
                    "interventions": [],
                    "halted": False,
                    "duration_ms": 1,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
