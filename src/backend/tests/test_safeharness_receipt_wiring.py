"""Enforcement-receipt wiring — SUMC side of the signed control-plane proofs.

The framework issues, binds, and verifies receipts (covered upstream in the
repo-root test suite). What was missing is the platform wiring: the runtime
built for chat/tool execution never received an issuer, and the audit sink
dropped the receipt on the floor, so a deployment had no tamper-evident proof
in its ledger even when the framework could have produced one. These tests
pin the wiring: env-driven issuer construction, countersigned execution, and
an audit row an offline verifier can actually check.
"""

import asyncio
import json
import os
import secrets
import sqlite3

import pytest
from types import SimpleNamespace

from safeharness import SafeHarnessRuntime, SessionContext, ToolDescriptor, ToolInvocation
from safeharness.adapters import execute_invocation
from safeharness.backends import (
    InMemoryEnforcementReceiptReplayStore,
    SignedCapabilityGateway,
    SignedEnforcementReceiptAuthority,
    StaticToolCatalog,
)
from safeharness.backends.receipt_config import (
    build_enforcement_receipt_issuer_from_environment,
)
from safeharness.runtime import RuntimeConfig, RuntimeServices

from core.safeharness_audit import SQLiteAuditSink, build_audit_sink
from core.safeharness_capability_mapping import build_internet_search_tool, build_local_runtime

_RECEIPT_ENV = [
    "SAFEHARNESS_ENFORCEMENT_RECEIPT_MODE",
    "SAFEHARNESS_ENFORCEMENT_RECEIPT_SIGNING_KEY",
    "SAFEHARNESS_ENFORCEMENT_RECEIPT_KEY_ID",
    "SAFEHARNESS_ENFORCEMENT_RECEIPT_PRIVATE_KEY_FILE",
    "SAFEHARNESS_ENFORCEMENT_RECEIPT_ISSUER",
]


def _clear_receipt_env(monkeypatch):
    for name in _RECEIPT_ENV:
        monkeypatch.delenv(name, raising=False)


def _hmac_env(monkeypatch, key: str = "unit-hmac-key-0123456789abcdef-0123456789abcdef"):
    _clear_receipt_env(monkeypatch)
    monkeypatch.setenv("SAFEHARNESS_ENFORCEMENT_RECEIPT_MODE", "hmac")
    monkeypatch.setenv("SAFEHARNESS_ENFORCEMENT_RECEIPT_SIGNING_KEY", key)


def test_local_runtime_has_no_issuer_without_config(monkeypatch):
    _clear_receipt_env(monkeypatch)
    runtime = build_local_runtime(build_internet_search_tool(), agent_id="agent-1")
    assert runtime.services.enforcement_receipt_issuer is None


def test_local_runtime_builds_hmac_issuer_from_env(monkeypatch):
    _hmac_env(monkeypatch)
    runtime = build_local_runtime(build_internet_search_tool(), agent_id="agent-1")
    issuer = runtime.services.enforcement_receipt_issuer
    assert isinstance(issuer, SignedEnforcementReceiptAuthority)


def test_local_runtime_refuses_misconfigured_receipt_env(monkeypatch):
    _hmac_env(monkeypatch)
    monkeypatch.setenv("SAFEHARNESS_ENFORCEMENT_RECEIPT_PRIVATE_KEY_FILE", "leak.pem")
    with pytest.raises(RuntimeError, match="enforcement receipt configuration invalid"):
        build_local_runtime(build_internet_search_tool(), agent_id="agent-1")


def _descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        "lookup",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        required_capabilities=frozenset({"lookup:read"}),
    )


def _runtime_with_receipts(tmp_path, monkeypatch, descriptor):
    _hmac_env(monkeypatch)
    key = os.environ["SAFEHARNESS_ENFORCEMENT_RECEIPT_SIGNING_KEY"]
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", str(tmp_path / "audit.db"))
    authority = SignedCapabilityGateway(
        secrets.token_bytes(32),
        {"agent-1": frozenset(descriptor.required_capabilities)},
    )
    runtime = SafeHarnessRuntime(
        config=RuntimeConfig(policy_version="safeharness.default.v1"),
        services=RuntimeServices(
            tool_catalog=StaticToolCatalog([descriptor]),
            capability_authority=authority,
            audit_sink=build_audit_sink(),
            enforcement_receipt_issuer=build_enforcement_receipt_issuer_from_environment(),
        ),
    )
    return runtime, authority, key


async def test_execute_invocation_countersigns_audits_and_verifies(tmp_path, monkeypatch):
    descriptor = _descriptor()
    runtime, capability_authority, key = _runtime_with_receipts(tmp_path, monkeypatch, descriptor)
    context = SessionContext(
        tenant_id="tenant-a",
        session_id="session-a",
        run_id="run-a",
        agent_id="agent-1",
        trace_id="trace-a",
        framework="sumc-receipt-wiring-test",
    )
    invocation = ToolInvocation(
        tool_name="lookup",
        arguments={"query": "alarm"},
        tool_version=descriptor.version,
    )
    grant = await capability_authority.issue(context, descriptor, invocation)
    observed = []

    async with runtime.session(context) as session:
        outcome = await execute_invocation(
            session,
            invocation,
            lambda effective: (observed.append(effective), {"ok": True})[1],
            capability_token=grant.token,
        )

    assert not outcome.blocked
    # The runtime countersigned the allowed execution; the effective
    # invocation carries the token.
    assert outcome.receipt is not None and outcome.receipt.enforced is True
    assert observed, "executor must observe the effective invocation"
    token = observed[0].enforcement_receipt

    # Offline verification: same HMAC material, untouched invocation. The
    # token arrives through the invocation, not the ledger (framework
    # invariant: the agent-side audit never persists receipt tokens).
    authority = SignedEnforcementReceiptAuthority(
        key, replay_store=InMemoryEnforcementReceiptReplayStore()
    )
    assert token, "expected the effective invocation to carry the countersigned receipt"
    verdict = await authority.verify(context, descriptor, invocation, token)
    assert verdict.allowed
    assert verdict.claims["tool_name"] == "lookup"

    # The ledger records that the execution was receipt-enforced, and it must
    # NOT contain the token itself.
    rows = sqlite3.connect(str(tmp_path / "audit.db")).execute(
        "SELECT record_type, event_kind, details FROM safeharness_audit "
        "WHERE record_type = 'enforcement'"
    ).fetchall()
    assert rows, "enforcement receipt must reach the audit ledger"
    ledger_blob = "".join(str(row) for row in rows)
    assert token not in ledger_blob
    metadata_rows = [
        json.loads(details_json or "{}").get("enforcement_receipt_record")
        for _rt, _kind, details_json in rows
        if json.loads(details_json or "{}").get("enforcement_receipt_record")
    ]
    assert metadata_rows, "receipt metadata (id/decision/enforced) must be auditable"


def test_audit_sink_persists_receipt_summary(tmp_path):
    from datetime import datetime, timedelta, timezone

    from safeharness.api.decisions import Disposition, EnforcementReceipt

    sink = SQLiteAuditSink(tmp_path / "audit.db")
    receipt = EnforcementReceipt(
        decision_id="decision-1",
        event_id="event-1",
        disposition=Disposition.ALLOW,
        enforced=True,
        adapter="generic-python",
        details={"phase": "post_tool", "enforcement_receipt": "shr1-token"},
        enforced_at=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    event = SimpleNamespace(
        kind=SimpleNamespace(value="tool_call_succeeded"),
        meta=SimpleNamespace(
            tenant_id="tenant-a",
            session_id="session-a",
            run_id="run-a",
            trace_id="trace-a",
            agent_id="agent-1",
            framework="sumc-audit-test",
            timestamp=datetime.now(timezone.utc),
        ),
        payload={"tool_name": "lookup"},
    )
    record = SimpleNamespace(
        record_type="enforcement",
        event=event,
        decision=SimpleNamespace(
            disposition=Disposition.ALLOW,
            reasons=[],
        ),
        receipt=receipt,
        details={},
    )
    asyncio.run(sink.write(record))

    row = sqlite3.connect(str(tmp_path / "audit.db")).execute(
        "SELECT record_type, decision_disposition, details FROM safeharness_audit "
        "WHERE record_type = 'enforcement'"
    ).fetchone()
    assert row is not None
    details = json.loads(row[2])
    merged = details["enforcement_receipt_record"]
    assert merged["receipt_id"] == receipt.receipt_id
    assert merged["decision_id"] == "decision-1"
    assert merged["enforced"] is True
    assert merged["details"]["enforcement_receipt"] == "shr1-token"
