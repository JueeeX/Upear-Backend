import asyncio

from orchestration.subagents.fsm_contracts import FSMPlanV1, default_agent_cards
from orchestration.subagents.fsm_controller import FSMController, FSMRuntime
from orchestration.subagents.fsm_validator import validate_fsm

from .test_fsm_validator import _valid_fsm


def _verified():
    result = validate_fsm(_valid_fsm(), default_agent_cards())
    assert result.valid and result.fsm is not None
    return result.fsm, result.agent_registry_hash


def test_controller_uses_declared_transitions_and_isolates_state_runs():
    fsm, registry_hash = _verified()
    calls = []

    async def handler(inputs, context):
        calls.append((context.state_id, context.session_id, context.run_id, context.delegation_id, dict(inputs)))
        if context.state_id == "query_before":
            return {"outcome": "success", "device_id": inputs["device_id"], "alarms": []}
        if context.state_id == "diagnose_before":
            return {"outcome": "resolved", "summary": "clear"}
        return {"outcome": "failure"}

    handlers = {
        ("noc-diagnosis-agent", "alarm.query"): handler,
        ("noc-diagnosis-agent", "alarm.diagnose"): handler,
    }
    events = asyncio.run(_collect(FSMController(fsm, parent_run_id="outer", agent_registry_hash=registry_hash, handlers=handlers).run()))
    assert events[-1]["type"] == "plan_complete"
    assert events[-1]["state_id"] == "complete"
    assert len(calls) == 2
    assert len({item[1] for item in calls}) == 2
    assert len({item[2] for item in calls}) == 2
    assert len({item[3] for item in calls}) == 2


def test_controller_pauses_for_approval_and_resumes_checkpoint():
    fsm, registry_hash = _verified()
    checkpoints = []

    async def handler(_inputs, _context):
        return {"outcome": "needs_remediation", "summary": "change needed"}

    handlers = {
        ("noc-diagnosis-agent", "alarm.query"): lambda inputs, _context: {"outcome": "success", "device_id": inputs["device_id"], "alarms": []},
        ("noc-diagnosis-agent", "alarm.diagnose"): handler,
    }
    first = FSMController(
        fsm,
        parent_run_id="outer",
        agent_registry_hash=registry_hash,
        handlers=handlers,
        checkpoint=lambda runtime: checkpoints.append(runtime.to_dict()),
    )
    events = asyncio.run(_collect(first.run()))
    assert events[-1]["type"] == "plan_approval_required"
    assert checkpoints[-1]["status"] == "awaiting_approval"
    resumed = FSMController(
        fsm,
        parent_run_id="outer",
        agent_registry_hash=registry_hash,
        handlers={
            **handlers,
            ("config-change-agent", "interface.restart"): lambda _inputs, _context: {"outcome": "success", "summary": "restarted"},
            ("noc-diagnosis-agent", "alarm.verify"): lambda _inputs, _context: {"outcome": "resolved", "summary": "clear"},
        },
        runtime={**checkpoints[-1], "status": "running"},
    )
    resumed_events = asyncio.run(_collect(resumed.run({"approve_restart": "approved"})))
    assert resumed_events[-1]["type"] == "plan_complete"


async def _collect(iterator):
    return [item async for item in iterator]


def test_controller_records_sanitized_error_class_on_failure():
    fsm, registry_hash = _verified()

    async def handler(_inputs, _context):
        raise RuntimeError("simulated tool failure with sensitive payload")

    handlers = {("noc-diagnosis-agent", "alarm.query"): handler}
    events = asyncio.run(
        _collect(FSMController(fsm, parent_run_id="outer", agent_registry_hash=registry_hash, handlers=handlers).run())
    )
    completed = [event for event in events if event["type"] == "plan_state_complete"]
    assert completed and completed[0]["output"]["error_class"] == "RuntimeError"
    assert completed[0]["output"]["error"] == "state execution failed"
    assert "sensitive payload" not in str(completed[0]["output"])
    assert events[-1]["type"] == "plan_complete"
