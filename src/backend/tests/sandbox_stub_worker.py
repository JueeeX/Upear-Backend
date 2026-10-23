"""Sandboxed runner pilot stub (P3.2 tests).

Speaks the sandbox worker stdio protocol but needs no framework library:
directives arrive via the request message ("sleep:<seconds>" delays the
response), and the response reply echoes the child's environment key NAMES
and the stdin-received api_key so tests can pin the isolation contract.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys


def main() -> int:
    request = json.loads(sys.stdin.read())
    message = str(request.get("message") or "")
    if message.startswith("sleep:"):
        asyncio.run(asyncio.sleep(float(message.split(":", 1)[1])))
    reply = (
        "ENV:" + ",".join(sorted(os.environ)) + ";KEY:" + str(request.get("api_key") or "")
    )
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
