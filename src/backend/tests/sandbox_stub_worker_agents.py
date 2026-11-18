"""Sandboxed runner stub for the agent-framework dispatch tests (P3.3).

Same stdio protocol as ``sandbox_stub_worker`` but echoes which framework the
parent requested (the new ``framework`` request field) plus a ``sleep:``
directive, proving per-framework dispatch reaches the child while keeping the
isolation contract visible (environment key NAMES + stdin-received api_key).
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
        "FW:" + str(request.get("framework") or "")
        + ";MSG:" + message
        + ";ENV:" + ",".join(sorted(os.environ))
        + ";KEY:" + str(request.get("api_key") or "")
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
