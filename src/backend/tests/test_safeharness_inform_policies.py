"""Platform policy construction helpers (safeharness_inform).

Pins the zero-change default: with no chains configured and no env thresholds,
``platform_extra_policies()`` returns an empty tuple, so the runtime policy
tuples at the construction sites are byte-identical to the historic ones.
"""

import pytest
from safeharness.policies.inform import InformPolicy
from safeharness.policies.rules_loader import clear_runtime_chains
from safeharness.policies.trajectory import TrajectoryPolicy

from core.safeharness_inform import (
    platform_extra_policies,
    platform_inform_and_extras,
    platform_inform_policy,
    platform_trajectory_policy,
)


@pytest.fixture(autouse=True)
def _clean_runtime_chains():
    clear_runtime_chains()
    yield
    clear_runtime_chains()


def test_extra_policies_empty_by_default(monkeypatch):
    monkeypatch.delenv("SAFEHARNESS_TRAJECTORY_RETRY_THRESHOLD", raising=False)
    monkeypatch.delenv("SAFEHARNESS_TRAJECTORY_ENTROPY", raising=False)
    assert platform_extra_policies() == ()
    inform, *extras = platform_inform_and_extras()
    assert isinstance(inform, InformPolicy)
    assert extras == []


def test_runtime_chains_activate_trajectory_policy():
    from safeharness.policies.rules_loader import apply_runtime_chains

    apply_runtime_chains([{"name": "c1", "first": "secret.*", "then": "http.*"}])
    extras = platform_extra_policies()
    assert len(extras) == 1
    assert isinstance(extras[0], TrajectoryPolicy)
    assert extras[0].configured is True
    assert extras[0]._chains[0].name == "c1"


def test_env_thresholds_activate_trajectory_policy(monkeypatch):
    monkeypatch.setenv("SAFEHARNESS_TRAJECTORY_ENTROPY", "5.5")
    extras = platform_extra_policies()
    assert len(extras) == 1
    assert extras[0].entropy_threshold == 5.5
    monkeypatch.setenv("SAFEHARNESS_TRAJECTORY_RETRY_THRESHOLD", "not-a-number")
    assert platform_trajectory_policy().repeated_attempt_threshold == 0  # bad env → off


def test_inform_layer_unchanged_by_chains():
    from safeharness.policies.rules_loader import apply_runtime_chains

    baseline = len(InformPolicy()._patterns)
    apply_runtime_chains([{"name": "c1", "first": "secret.*", "then": "http.*"}])
    inform = platform_inform_policy()
    assert len(inform._patterns) == baseline  # chains never touch the inform layer


def test_runtime_chain_tags_flow_into_trajectory_policy():
    from safeharness.policies.rules_loader import apply_runtime_chains

    apply_runtime_chains(
        [
            {
                "name": "tagged",
                "first_tags": ["secret"],
                "then_tags": ["network", "upload"],
            }
        ]
    )
    policy = platform_trajectory_policy()
    assert policy.configured is True
    chain = policy._chains[0]
    assert chain.first_tags == ("secret",)
    assert chain.then_tags == ("network", "upload")
    assert chain.first == "" and chain.then == ""
