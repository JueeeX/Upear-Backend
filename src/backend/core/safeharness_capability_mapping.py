"""Explicit SafeHarness mappings for capabilities with a verified runtime path."""

from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field

INTERNET_SEARCH_ID = "internet_search"
INTERNET_SEARCH_CAPABILITY = "internet:search"
POLICY_PACKAGE_ID = "safeharness.default.v1"


async def _protected_internet_search(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    max_results: Annotated[int, Field(ge=1, le=10)] = 5,
) -> dict[str, Any]:
    """Search public web sources when authoritative internal sources are insufficient."""
    from mcp_servers.internet_search_mcp.server import internet_search

    return await internet_search(
        query=query,
        max_results=max_results,
        topic="general",
        search_depth="advanced",
        include_raw_content=False,
        cn_only=True,
    )


@lru_cache(maxsize=1)
def build_internet_search_tool():
    """Build the one mapped tool from the existing MCP callable."""
    from safeharness.adapters.langgraph import LangGraphTool
    from safeharness.api.tools import Reversibility

    return LangGraphTool.from_callable(
        _protected_internet_search,
        name=INTERNET_SEARCH_ID,
        required_capabilities={INTERNET_SEARCH_CAPABILITY},
        version="1",
        side_effects={"network_egress"},
        data_classifications={"public_web", "user_query"},
        reversibility=Reversibility.READ_ONLY,
        idempotent=True,
        sandbox_required=False,
    )


def capability_mapping(identifier: object, *, kind: str) -> dict[str, Any] | None:
    if kind != "mcp" or str(identifier or "").strip() != INTERNET_SEARCH_ID:
        return None
    try:
        descriptor = build_internet_search_tool().descriptor
    except (ImportError, TypeError, ValueError):
        return None
    return {
        "tool_descriptor": {
            "status": "mapped",
            "name": descriptor.name,
            "version": descriptor.version,
            "input_schema": dict(descriptor.input_schema),
            "required_capabilities": sorted(descriptor.required_capabilities),
            "side_effects": sorted(descriptor.side_effects),
            "data_classifications": sorted(descriptor.data_classifications),
            "reversibility": descriptor.reversibility.value,
            "idempotent": descriptor.idempotent,
            "compensator": descriptor.compensator,
            "sandbox_required": descriptor.sandbox_required,
            "integrity_digest": descriptor.integrity_digest,
        },
        "adapter_coverage": {
            "status": "mapped",
            "framework": "langgraph",
            "boundaries": [
                "input_filter",
                "pre_tool_enforcement",
                "tool_argument_rewrite",
                "post_tool_filter",
                "final_output_filter",
                "message_interception",
            ],
        },
        "policy_package": {
            "status": "mapped",
            "id": POLICY_PACKAGE_ID,
            "required_capability": INTERNET_SEARCH_CAPABILITY,
        },
        "execution": {
            "status": "approval_required",
            "reason_code": "safeharness_one_shot_approval_required",
            "production_ready": False,
        },
    }


def build_local_runtime(tool, *, agent_id: str, approval_broker=None, judge_provider=None, capability_authority=None):
    """Return a per-run local reference runtime; remote Gateway stays unclaimed.

    When the deployment configures enforcement-receipt signing (HMAC key or
    Ed25519 key file via ``SAFEHARNESS_ENFORCEMENT_RECEIPT_*``), every allowed
    tool execution through this runtime is countersigned by the control plane
    and the receipt lands in the audit trail. Unconfigured deployments keep the
    previous behaviour (no receipts) — the environment builder returns None.

    ``capability_authority`` replaces the default local random-key gateway for
    the policy-side capability checks (gateway-routed deployments pass one so
    ConstrainPolicy verifies gateway-signed tokens).
    """
    from core.safeharness_audit import build_audit_sink
    from safeharness import SafeHarnessRuntime
    from safeharness.backends import SignedCapabilityGateway, StaticToolCatalog
    from safeharness.backends.receipt_config import (
        build_enforcement_receipt_issuer_from_environment,
    )
    from safeharness.runtime import RuntimeConfig, RuntimeServices

    catalog = StaticToolCatalog([tool.descriptor])
    authority = capability_authority
    if authority is None:
        authority = SignedCapabilityGateway(
            secrets.token_bytes(32),
            {agent_id: {INTERNET_SEARCH_CAPABILITY}},
        )
    try:
        enforcement_receipt_issuer = build_enforcement_receipt_issuer_from_environment()
    except ValueError as exc:
        # A misconfigured signing setup must not silently downgrade to unsigned
        # executions: refuse to build the runtime instead.
        raise RuntimeError(f"enforcement receipt configuration invalid: {exc}") from exc
    return SafeHarnessRuntime(
        config=RuntimeConfig(policy_version=POLICY_PACKAGE_ID),
        services=RuntimeServices(
            tool_catalog=catalog,
            capability_authority=authority,
            audit_sink=build_audit_sink(),
            approval_broker=approval_broker,
            judge_provider=judge_provider,
            enforcement_receipt_issuer=enforcement_receipt_issuer,
        ),
    )
