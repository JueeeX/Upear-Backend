"""Process-local Alpha coordinator for controlled SafeHarness mutations."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from core.auth.password import verify_password
from core.db.models import LocalUser
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from sqlalchemy.orm import Session

IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$"
POLICY_VERSION = "safeharness.mutations.v1"
APPROVAL_TTL_SECONDS = 300


class MutationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tenant_id: str = Field(min_length=1, max_length=256, pattern=IDENTIFIER_PATTERN)
    agent_id: str | None = Field(default=None, min_length=1, max_length=256, pattern=IDENTIFIER_PATTERN)
    delegation_id: str | None = Field(default=None, min_length=1, max_length=256, pattern=IDENTIFIER_PATTERN)
    kind: Literal["skill", "agent", "mcp"] | None = None
    item_id: str | None = Field(default=None, min_length=1, max_length=256, pattern=IDENTIFIER_PATTERN)
    install_id: str | None = Field(default=None, min_length=1, max_length=256, pattern=IDENTIFIER_PATTERN)
    task_id: str | None = Field(default=None, min_length=1, max_length=256, pattern=IDENTIFIER_PATTERN)


class MutationReauth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["password"]
    credential: SecretStr = Field(min_length=1, max_length=512)


class MutationPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"]
    action: Literal[
        "agent.quarantine",
        "agent.restore",
        "delegation.revoke",
        "capability.enable",
        "capability.disable",
        "plugin.enable",
        "plugin.disable",
        "automation.run",
    ]
    target: MutationTarget
    reason: str = Field(min_length=3, max_length=500)
    idempotency_key: str = Field(min_length=16, max_length=128, pattern=IDENTIFIER_PATTERN)
    reauth: MutationReauth

    @model_validator(mode="after")
    def validate_target(self) -> "MutationPrepareRequest":
        expected = {
            "agent.quarantine": {"agent_id"},
            "agent.restore": {"agent_id"},
            "delegation.revoke": {"delegation_id"},
            "capability.enable": {"kind", "item_id"},
            "capability.disable": {"kind", "item_id"},
            "plugin.enable": {"install_id"},
            "plugin.disable": {"install_id"},
            "automation.run": {"task_id"},
        }[self.action]
        present = {
            key
            for key in ("agent_id", "delegation_id", "kind", "item_id", "install_id", "task_id")
            if getattr(self.target, key) is not None
        }
        if present != expected:
            fields = ", ".join(f"target.{key}" for key in sorted(expected))
            raise ValueError(f"{self.action} requires only {fields}")
        return self


class MutationDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"]
    decision: Literal["allow", "deny"]
    idempotency_key: str = Field(min_length=16, max_length=128, pattern=IDENTIFIER_PATTERN)


@dataclass
class PendingMutation:
    approval_id: str
    decision_id: str
    operator_id: str
    tenant_id: str
    action: str
    target_values: dict[str, str]
    reason: str
    idempotency_key: str
    request_digest: str
    expires_at: float
    state: str = "pending"
    result: dict[str, Any] | None = None

    @property
    def resource(self) -> str:
        kind, identifier = {
            "agent.quarantine": ("agent", self.target_values.get("agent_id")),
            "agent.restore": ("agent", self.target_values.get("agent_id")),
            "delegation.revoke": ("delegation", self.target_values.get("delegation_id")),
            "capability.enable": (f"capability/{self.target_values.get('kind')}", self.target_values.get("item_id")),
            "capability.disable": (f"capability/{self.target_values.get('kind')}", self.target_values.get("item_id")),
            "plugin.enable": ("plugin", self.target_values.get("install_id")),
            "plugin.disable": ("plugin", self.target_values.get("install_id")),
            "automation.run": ("automation", self.target_values.get("task_id")),
        }[self.action]
        return f"tenant/{self.tenant_id}/{kind}/{identifier}"

    @property
    def effect(self) -> str:
        return {
            "agent.quarantine": "deny_agent_communication",
            "agent.restore": "restore_agent_communication",
            "delegation.revoke": "revoke_delegated_authority",
            "capability.enable": "enable_capability",
            "capability.disable": "disable_capability",
            "plugin.enable": "enable_plugin",
            "plugin.disable": "disable_plugin",
            "automation.run": "schedule_automation_run",
        }[self.action]

    @property
    def target(self) -> dict[str, str]:
        return {"tenant_id": self.tenant_id, **self.target_values}


class MutationConflict(RuntimeError):
    pass


class MutationUnavailable(RuntimeError):
    pass


class MutationIndeterminate(RuntimeError):
    pass


class MutationCoordinator:
    """One-shot target-bound approval and idempotency state.

    ponytail: process-local state is Alpha-only; replace with an atomic shared
    store before multi-worker or production deployment.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_approval: dict[str, PendingMutation] = {}
        self._by_idempotency: dict[tuple[str, str, str], PendingMutation] = {}

    def prepare(
        self,
        operation: MutationPrepareRequest,
        *,
        operator_id: str,
        authoritative_tenant_id: str,
        db: Session,
    ) -> tuple[PendingMutation, bool]:
        if operation.target.tenant_id != authoritative_tenant_id:
            raise PermissionError("cross-tenant mutation")
        if not _verify_local_reauth(db, operator_id, operation.reauth):
            raise PermissionError("reauthentication failed")

        digest = _request_digest(operation, operator_id)
        key = (authoritative_tenant_id, operator_id, operation.idempotency_key)
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            existing = self._by_idempotency.get(key)
            if existing is not None:
                if existing.request_digest != digest:
                    raise MutationConflict("idempotency key is bound to another mutation")
                return existing, False

            item = PendingMutation(
                approval_id=uuid.uuid4().hex,
                decision_id=str(uuid.uuid4()),
                operator_id=operator_id,
                tenant_id=authoritative_tenant_id,
                action=operation.action,
                target_values={
                    key: value
                    for key, value in operation.target.model_dump(exclude={"tenant_id"}).items()
                    if value is not None
                },
                reason=operation.reason,
                idempotency_key=operation.idempotency_key,
                request_digest=digest,
                expires_at=now + APPROVAL_TTL_SECONDS,
            )
            self._by_approval[item.approval_id] = item
            self._by_idempotency[key] = item
            return item, True

    def begin_decision(
        self,
        approval_id: str,
        decision: MutationDecisionRequest,
        *,
        operator_id: str,
        authoritative_tenant_id: str,
    ) -> tuple[PendingMutation, bool]:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            item = self._by_approval.get(approval_id)
            if item is None:
                raise MutationUnavailable("approval is missing or expired")
            if (
                item.operator_id != operator_id
                or item.tenant_id != authoritative_tenant_id
                or item.idempotency_key != decision.idempotency_key
            ):
                raise PermissionError("approval context mismatch")
            if item.state == "completed" and item.result is not None:
                return item, False
            if item.state == "indeterminate":
                raise MutationIndeterminate("mutation outcome requires reconciliation")
            if item.state != "pending":
                raise MutationConflict("mutation is already being processed")
            if decision.decision == "deny":
                item.state = "denied"
                item.result = _decision_payload(item, status="denied")
                return item, False
            item.state = "executing"
            return item, True

    def complete(self, item: PendingMutation, result: dict[str, Any]) -> None:
        with self._lock:
            if item.state != "executing":
                raise MutationConflict("mutation is not executing")
            item.state = "completed"
            item.result = result

    def indeterminate(self, item: PendingMutation) -> None:
        with self._lock:
            if item.state == "executing":
                item.state = "indeterminate"

    def reset_for_tests(self) -> None:
        with self._lock:
            self._by_approval.clear()
            self._by_idempotency.clear()

    def _prune(self, now: float) -> None:
        expired = [
            approval_id
            for approval_id, item in self._by_approval.items()
            if item.expires_at <= now and item.state == "pending"
        ]
        for approval_id in expired:
            item = self._by_approval.pop(approval_id)
            self._by_idempotency.pop(
                (item.tenant_id, item.operator_id, item.idempotency_key), None
            )


def _verify_local_reauth(db: Session, user_id: str, reauth: MutationReauth) -> bool:
    try:
        local = db.get(LocalUser, user_id)
        return bool(
            local
            and local.status == "active"
            and verify_password(reauth.credential.get_secret_value(), local.password_hash)
        )
    except Exception:  # noqa: BLE001 - unavailable authority must fail closed
        return False


def _request_digest(operation: MutationPrepareRequest, operator_id: str) -> str:
    value = {
        "schema_version": operation.schema_version,
        "action": operation.action,
        "target": operation.target.model_dump(),
        "reason": operation.reason,
        "idempotency_key": operation.idempotency_key,
        "operator_id": operator_id,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _decision_payload(item: PendingMutation, *, status: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": status,
        "approval_id": item.approval_id,
        "decision_id": item.decision_id,
        "policy_version": POLICY_VERSION,
        "action": item.action,
        "target": item.target,
        "resource": item.resource,
        "effect": item.effect,
        "operator_id": item.operator_id,
        "reason": item.reason,
        "idempotency_key": item.idempotency_key,
    }


mutation_coordinator = MutationCoordinator()


__all__ = [
    "APPROVAL_TTL_SECONDS",
    "MutationConflict",
    "MutationDecisionRequest",
    "MutationIndeterminate",
    "MutationPrepareRequest",
    "MutationUnavailable",
    "PendingMutation",
    "POLICY_VERSION",
    "_decision_payload",
    "mutation_coordinator",
]
