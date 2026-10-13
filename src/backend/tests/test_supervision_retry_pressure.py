"""Supervision trigger 5: retry-pressure (AgentGuard repeated-attempts pattern).

Pins the additive contract: ``retry_trigger=0`` (the default and every stored
config from before this field existed) keeps the four historic triggers as the
entire behaviour; enabling the trigger only ever adds interventions. Also pins
the gateway-tool denial feed into the supervisor sink.
"""

from core.external_agents.gateway_tools import execute_gateway_tool
from core.external_agents.supervisor import (
    DEFAULT_SUPERVISION,
    SupervisionConfig,
    SupervisionState,
    load_supervision_config,
    save_supervision_config,
)
from safeharness.gateway.client import ToolGatewayError


class _FakeRow:
    def __init__(self, config_value: str = "") -> None:
        self.config_value = config_value


class _FakeDB:
    def __init__(self, row=None):
        self.row = row

    def get(self, model, key):
        return self.row

    def add(self, obj):
        self.row = obj

    def commit(self):
        return None


def _state(**overrides) -> SupervisionState:
    config = SupervisionConfig(**{"max_steps": 99, **overrides})
    return SupervisionState(config=config)


def test_default_config_has_retry_trigger_disabled():
    assert DEFAULT_SUPERVISION.retry_trigger == 0
    assert SupervisionConfig().to_dict()["retry_trigger"] == 0


def test_disabled_trigger_ignores_denials():
    state = _state()
    for _ in range(10):
        state.record_tool_denial("internet_search")
    for i in range(5):
        assert state.check(output_text=f"step {i}") is None
    assert state.interventions == []


def test_retry_pressure_fires_at_threshold():
    state = _state(retry_trigger=3)
    state.record_tool_denial("internet_search")
    state.record_tool_denial("internet_search")
    assert state.check(output_text="step 1") is None
    state.record_tool_denial("internet_search")
    intervention = state.check(output_text="step 2")
    assert intervention is not None
    assert intervention["trigger"] == "retry_pressure"
    assert "internet_search" in intervention["detail"]
    # Re-fires while the counter stays over threshold, same semantics as
    # max_steps/total_budget (the registry halts at the first intervention).
    again = state.check(output_text="step 3")
    assert again is not None and again["trigger"] == "retry_pressure"


def test_retry_pressure_counts_per_tool():
    state = _state(retry_trigger=3)
    state.record_tool_denial("a")
    state.record_tool_denial("a")
    state.record_tool_denial("b")
    state.record_tool_denial("b")
    assert state.check(output_text="step 1") is None
    state.record_tool_denial("b")
    intervention = state.check(output_text="step 2")
    assert intervention is not None and "b" in intervention["detail"]


def test_historic_triggers_still_win_when_both_match():
    state = _state(retry_trigger=1, max_steps=1)
    assert state.check(output_text="warm up") is None  # steps=1, cap not crossed
    state.record_tool_denial("a")
    intervention = state.check(output_text="step 2")
    assert intervention["trigger"] == "max_steps"


def test_config_roundtrip_with_and_without_retry_trigger():
    import json

    db = _FakeDB()
    save_supervision_config(db, SupervisionConfig(retry_trigger=4))
    stored = db.row.config_value
    assert load_supervision_config(_FakeDB(_FakeRow(stored))).retry_trigger == 4
    # A pre-existing row written before the field existed loads disabled.
    raw = json.loads(stored)
    legacy = json.dumps(
        {k: v for k, v in raw.items() if k != "retry_trigger"}, ensure_ascii=False
    )
    assert "retry_trigger" not in legacy
    assert load_supervision_config(_FakeDB(_FakeRow(legacy))).retry_trigger == 0


class _StubIssuer:
    async def issue(self, *args, **kwargs):
        return "receipt-token"


class _DenyingClient:
    async def issue(self, *args, **kwargs):
        return type("Grant", (), {"token": "capability-token"})()

    async def execute(self, *args, **kwargs):
        raise ToolGatewayError("capability_denied", "denied by gateway")


async def test_gateway_denial_feeds_supervision_sink():
    from core.safeharness_tool_gateway import _demo_tool
    from safeharness.api.context import SessionContext

    denials: list[tuple[str, str]] = []
    bundle = {
        "tool_name": "system_status_snapshot",
        "client": _DenyingClient(),
        "issuer": _StubIssuer(),
        "context": SessionContext(
            tenant_id="t",
            agent_id="a",
            session_id="s",
            run_id="r",
        ),
        "descriptor": _demo_tool().descriptor,
        "calls": [],
        "on_denial": lambda tool, code: denials.append((tool, code)),
    }
    result = await execute_gateway_tool(bundle)
    assert "error" in result
    assert bundle["calls"][-1]["ok"] is False
    assert denials == [("system_status_snapshot", "capability_denied")]


async def test_gateway_without_sink_unchanged():
    from core.safeharness_tool_gateway import _demo_tool
    from safeharness.api.context import SessionContext

    bundle = {
        "tool_name": "system_status_snapshot",
        "client": _DenyingClient(),
        "issuer": _StubIssuer(),
        "context": SessionContext(tenant_id="t", agent_id="a", session_id="s", run_id="r"),
        "descriptor": _demo_tool().descriptor,
        "calls": [],
    }
    result = await execute_gateway_tool(bundle)
    assert "error" in result  # no sink, no crash — historic contract
