"""Central Tool Gateway assembly — the platform side of the authorisation split.

The framework ships the whole gateway (capability issuance, enforcement-receipt
verification with persistent replay protection, a fail-closed tool registry
whose executors never leave the server, and an HTTP transport). What the
platform lacked is the assembly: the demo ran every tool through a per-request
local runtime, so "authorisation" and "execution" lived in the agent process
and the enforcement receipts issued there had no consumer.

This module is the single place that:

* builds the **server** application (``build_gateway_application``) — the tool
  registry carries the real executors, capability signing lives only here, and
  every execution requires a control-plane enforcement receipt;
* runs it (``run_gateway_server``) as a standalone loopback service;
* hands the **agent** side a client (``gateway_client_from_environment``) so a
  runtime can issue capabilities and route executions through the gateway
  instead of executing locally.

Configuration is environment-driven and mirrors the receipt setup: without
``SAFEHARNESS_TOOL_GATEWAY_URL`` the agent keeps its local runtime; a
half-configured gateway deployment fails loudly instead of silently downgrading
to unsigned local execution.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

GATEWAY_DEFAULT_PORT = 38310
GATEWAY_DEFAULT_ISSUER_TENANT = "fiberhome-demo"
GATEWAY_DEFAULT_ISSUER_AGENT = "main_agent"

_ENV_URL = "SAFEHARNESS_TOOL_GATEWAY_URL"
_ENV_API_KEY = "SAFEHARNESS_TOOL_GATEWAY_API_KEY"
_ENV_HOST = "SAFEHARNESS_TOOL_GATEWAY_HOST"
_ENV_PORT = "SAFEHARNESS_TOOL_GATEWAY_PORT"
_ENV_CAPABILITY_KEY = "SAFEHARNESS_TOOL_GATEWAY_CAPABILITY_KEY"
_ENV_REPLAY_DB = "SAFEHARNESS_TOOL_GATEWAY_REPLAY_DB"


def gateway_client_from_environment(environ: dict[str, str] | None = None):
    """Agent-side client, or None when no gateway deployment is configured."""
    values = os.environ if environ is None else environ
    base_url = (values.get(_ENV_URL) or "").strip()
    api_key = (values.get(_ENV_API_KEY) or "").strip()
    if not base_url:
        return None
    from safeharness.gateway.client import RemoteToolGatewayClient

    return RemoteToolGatewayClient(base_url, api_key)


def _internet_search_executor():
    """The real executor, registered server-side only."""
    from core.safeharness_capability_mapping import _protected_internet_search

    return _protected_internet_search


GATEWAY_DEMO_TOOL_ID = "system_status_snapshot"
GATEWAY_DEMO_TOOL_CAPABILITY = "platform:status:read"


async def _system_status_executor() -> dict[str, Any]:
    """Safe read-only demo tool: process-local facts only — no files, no network."""
    import platform as _platform
    import sys as _sys
    from datetime import datetime, timezone

    return {
        "utc_now": datetime.now(timezone.utc).isoformat(),
        "python_version": _sys.version.split()[0],
        "platform": _platform.system(),
        "pid": os.getpid(),
    }


def _demo_tool():
    """Descriptor for the read-only demo tool (shared with the agent-side bridge)."""
    from safeharness.adapters.langgraph import LangGraphTool
    from safeharness.api.tools import Reversibility

    return LangGraphTool.from_callable(
        _system_status_executor,
        name=GATEWAY_DEMO_TOOL_ID,
        description=(
            "Return a read-only snapshot of the platform runtime status "
            "(UTC time, Python version, OS family, server process id). "
            "Use it to report live platform health facts."
        ),
        required_capabilities={GATEWAY_DEMO_TOOL_CAPABILITY},
        version="1",
        side_effects=frozenset(),
        data_classifications={"system_metadata"},
        reversibility=Reversibility.READ_ONLY,
        idempotent=True,
        sandbox_required=False,
    )


def _replay_db_path(values: dict[str, str]) -> str:
    explicit = (values.get(_ENV_REPLAY_DB) or "").strip()
    if explicit:
        return explicit
    from core.safeharness_audit import resolve_audit_db_path

    audit_path = resolve_audit_db_path()
    if audit_path:
        from pathlib import Path

        return str(Path(audit_path).with_name("safeharness_receipt_replay.db"))
    return "safeharness_receipt_replay.db"


class SQLiteGatewayAuditSink:
    """Persist ``GatewayAuditRecord`` facts next to the platform audit ledger.

    The gateway is the receipt-verification authority; without its own ledger
    the verification side of the receipt system would leave no trace. Writes go
    to a dedicated table in the same SQLite database as the platform audit
    records, via a worker thread (the platform sink's durability discipline).
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS safeharness_gateway_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id TEXT NOT NULL,
        action TEXT NOT NULL,
        outcome TEXT NOT NULL,
        tenant_id TEXT DEFAULT '',
        agent_id TEXT DEFAULT '',
        session_id TEXT DEFAULT '',
        run_id TEXT DEFAULT '',
        trace_id TEXT DEFAULT '',
        tool_name TEXT DEFAULT '',
        tool_version TEXT DEFAULT '',
        invocation_digest TEXT DEFAULT '',
        capability_fingerprint TEXT DEFAULT '',
        reason_code TEXT DEFAULT '',
        request_id TEXT DEFAULT '',
        details TEXT DEFAULT '{}',
        occurred_at TEXT NOT NULL,
        written_at TEXT NOT NULL
    )
    """

    def __init__(self, db_path: str):
        from pathlib import Path

        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(path)
        self._init_lock = __import__("threading").Lock()
        self._initialised = False

    def _connection(self):
        import sqlite3

        conn = sqlite3.connect(self._path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self) -> None:
        with self._init_lock:
            if self._initialised:
                return
            conn = self._connection()
            try:
                conn.execute(self._DDL)
                conn.commit()
            finally:
                conn.close()
            self._initialised = True

    def _write_sync(self, record: Any) -> None:
        import json
        from datetime import datetime, timezone

        self._ensure_schema()
        conn = self._connection()
        try:
            conn.execute(
                "INSERT INTO safeharness_gateway_audit ("
                " record_id, action, outcome, tenant_id, agent_id, session_id, run_id,"
                " trace_id, tool_name, tool_version, invocation_digest,"
                " capability_fingerprint, reason_code, request_id, details,"
                " occurred_at, written_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.record_id,
                    record.action,
                    record.outcome,
                    record.tenant_id,
                    record.agent_id,
                    record.session_id,
                    record.run_id,
                    record.trace_id,
                    record.tool_name,
                    record.tool_version,
                    record.invocation_digest,
                    record.capability_fingerprint,
                    record.reason_code,
                    record.request_id,
                    json.dumps(dict(record.details), ensure_ascii=False, default=str),
                    record.occurred_at.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    async def write_gateway(self, record: Any) -> None:
        import asyncio

        await asyncio.to_thread(self._write_sync, record)


def _gateway_audit_db_path(values: dict[str, str]) -> str:
    explicit = (values.get("SAFEHARNESS_TOOL_GATEWAY_AUDIT_DB") or "").strip()
    if explicit:
        return explicit
    from core.safeharness_audit import resolve_audit_db_path

    audit_path = resolve_audit_db_path()
    if audit_path:
        return audit_path
    return "safeharness_gateway_audit.db"


def build_gateway_application(environ: dict[str, str] | None = None):
    """Server application: real executors, receipt-mandatory, replay-persistent.

    Capability signing material is generated here (server-only) and never
    reaches agent processes; execution requires both a gateway capability
    token and a valid, unreplayed control-plane enforcement receipt.
    """
    values = os.environ if environ is None else environ
    capability_key = (values.get(_ENV_CAPABILITY_KEY) or "").strip()
    if len(capability_key) < 32:
        raise ValueError(
            f"{_ENV_CAPABILITY_KEY} must contain at least 32 characters"
        )
    from safeharness.backends import (
        SignedCapabilityGateway,
        SQLiteEnforcementReceiptReplayStore,
    )
    from safeharness.backends.receipt_config import (
        build_enforcement_receipt_verifier_from_environment,
    )
    from safeharness.gateway.service import (
        RegisteredTool,
        ToolGatewayApplication,
        ToolRegistry,
    )

    from core.safeharness_capability_mapping import build_internet_search_tool

    search_tool = build_internet_search_tool()
    demo = _demo_tool()
    verifier = build_enforcement_receipt_verifier_from_environment(
        replay_store=SQLiteEnforcementReceiptReplayStore(_replay_db_path(values)),
        environ=values,
    )
    if verifier is None:
        raise ValueError(
            "gateway requires enforcement-receipt verification: set "
            "SAFEHARNESS_ENFORCEMENT_RECEIPT_MODE=hmac with "
            "SAFEHARNESS_ENFORCEMENT_RECEIPT_SIGNING_KEY, or the Ed25519 "
            "public-key files"
        )
    application = ToolGatewayApplication(
        registry=ToolRegistry(
            [
                RegisteredTool(search_tool.descriptor, _internet_search_executor()),
                RegisteredTool(demo.descriptor, _system_status_executor),
            ]
        ),
        capability_gateway=SignedCapabilityGateway(
            capability_key.encode("utf-8"),
            {
                (
                    values.get("SAFEHARNESS_DEMO_TENANT_ID", GATEWAY_DEFAULT_ISSUER_TENANT),
                    values.get("SAFEHARNESS_COMMUNICATION_AGENT_ID", GATEWAY_DEFAULT_ISSUER_AGENT),
                ): frozenset(
                    search_tool.descriptor.required_capabilities
                    | demo.descriptor.required_capabilities
                ),
            },
        ),
        enforcement_receipt_verifier=verifier,
        require_enforcement_receipt=True,
        audit_sink=SQLiteGatewayAuditSink(_gateway_audit_db_path(values)),
    )
    return application


def run_gateway_server() -> int:
    """Blocking entry point for the standalone gateway process. Never returns
    normally while serving; returns the process exit code on failure."""
    host = (os.environ.get(_ENV_HOST) or "127.0.0.1").strip()
    try:
        port = int((os.environ.get(_ENV_PORT) or "").strip() or GATEWAY_DEFAULT_PORT)
    except ValueError:
        port = GATEWAY_DEFAULT_PORT
    api_key = (os.environ.get(_ENV_API_KEY) or "").strip()
    if len(api_key) < 32:
        logger.error("[tool-gateway] %s must contain at least 32 characters", _ENV_API_KEY)
        return 2
    try:
        application = build_gateway_application()
    except ValueError as exc:
        logger.error("[tool-gateway] configuration invalid: %s", exc)
        return 2
    from safeharness.gateway.service import create_gateway_server

    server = create_gateway_server(
        application,
        host=host,
        port=port,
        api_key=api_key,
        tenant_id=os.environ.get("SAFEHARNESS_DEMO_TENANT_ID", GATEWAY_DEFAULT_ISSUER_TENANT),
        agent_id=os.environ.get("SAFEHARNESS_COMMUNICATION_AGENT_ID", GATEWAY_DEFAULT_ISSUER_AGENT),
    )
    logger.info("[tool-gateway] listening on http://%s:%s", host, server.server_address[1])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = [
    "GATEWAY_DEFAULT_PORT",
    "GATEWAY_DEMO_TOOL_CAPABILITY",
    "GATEWAY_DEMO_TOOL_ID",
    "build_gateway_application",
    "gateway_client_from_environment",
    "run_gateway_server",
]
