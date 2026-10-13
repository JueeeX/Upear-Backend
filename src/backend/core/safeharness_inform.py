"""Platform InformPolicy / trajectory construction point.

``build_inform_policy()`` (framework) already merges builtin defaults + file
rules + runtime rules. This wrapper adds one process-level DB resync so
custom rules stored in system_configs survive a backend restart: the first
construction after process start re-applies the stored state, then the flag
short-circuits (zero DB cost on the per-message hot path afterwards).

Trajectory policies (cross-tool chain rules, repeated-attempt pressure,
entropy payloads) follow the same additive contract: ``platform_extra_policies()``
returns an empty tuple unless chains are configured or the env thresholds are
set, so wiring it into the runtime policy tuples changes nothing by default.
"""

from __future__ import annotations

import os

from safeharness.policies.rules_loader import (
    ENV_RULES_DIR,
    load_chains_from_dir,
    runtime_chains,
)
from safeharness.policies.trajectory import ChainRule, TrajectoryPolicy
from safeharness.policies.rules_loader import build_inform_policy

__all__ = [
    "platform_extra_policies",
    "platform_inform_and_extras",
    "platform_inform_policy",
    "platform_trajectory_policy",
]

_synced = False


def ensure_runtime_rules_synced() -> None:
    global _synced
    if _synced:
        return
    _synced = True
    try:
        from core.db.engine import SessionLocal
        from core.safeharness_policy_rules import resync_runtime_rules

        with SessionLocal() as db:
            resync_runtime_rules(db)
    except Exception:  # noqa: BLE001 — resync must never break the chat path
        pass


def platform_inform_policy():
    ensure_runtime_rules_synced()
    return build_inform_policy()


def _env_int(name: str) -> int:
    try:
        return max(0, int(os.environ.get(name, "").strip() or 0))
    except ValueError:
        return 0


def _env_float(name: str) -> float:
    try:
        value = float(os.environ.get(name, "").strip() or 0)
    except ValueError:
        return 0.0
    return value if value > 0 else 0.0


def platform_trajectory_policy() -> TrajectoryPolicy:
    """TrajectoryPolicy fed from file chains + runtime chains + env thresholds."""
    ensure_runtime_rules_synced()
    chains = []
    directory = (os.environ.get(ENV_RULES_DIR) or "").strip()
    if directory:
        try:
            chains.extend(load_chains_from_dir(directory))
        except Exception:  # noqa: BLE001 — build_inform_policy reports the same error
            pass
    chains.extend(runtime_chains())
    return TrajectoryPolicy(
        tuple(
            ChainRule(
                name=chain.name,
                first=chain.first,
                then=chain.then,
                severity=chain.severity,
                category=chain.category,
                reason=chain.reason,
                max_gap=chain.max_gap,
                first_tags=chain.first_tags,
                then_tags=chain.then_tags,
            )
            for chain in chains
        ),
        repeated_attempt_threshold=_env_int("SAFEHARNESS_TRAJECTORY_RETRY_THRESHOLD"),
        entropy_threshold=_env_float("SAFEHARNESS_TRAJECTORY_ENTROPY"),
    )


def platform_extra_policies() -> tuple:
    """Extra runtime policies; empty (zero change) unless configured."""
    trajectory = platform_trajectory_policy()
    return (trajectory,) if trajectory.configured else ()


def platform_inform_and_extras() -> tuple:
    return (platform_inform_policy(), *platform_extra_policies())
