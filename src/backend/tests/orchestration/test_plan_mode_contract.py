import asyncio
from types import SimpleNamespace

from orchestration.subagents.plan_mode import (
    _plan_error_event,
    _parse_plan_json,
    _validate_plan_candidate,
    astream_generate_plan,
)
from orchestration.subagents.fsm_contracts import default_agent_cards
from core.services.plan_service import PlanService
from safeharness.ports import JudgeResult


def test_plan_candidate_requires_normal_plan_shape():
    valid = {
        "title": "检查链路",
        "description": "收集状态并输出结论",
        "steps": [{"title": "收集状态", "description": "读取设备状态"}],
    }

    assert _validate_plan_candidate(valid, fsm_mode=False, fsm_registry=[]) == (None, None, None)
    code, _, _ = _validate_plan_candidate(
        {"title": "检查链路", "description": "收集状态", "steps": []},
        fsm_mode=False,
        fsm_registry=[],
    )
    assert code == "plan_json_invalid"


def test_fsm_candidate_requires_only_top_level_fsm():
    code, _, _ = _validate_plan_candidate(
        {"title": "旧格式", "fsm": {}},
        fsm_mode=True,
        fsm_registry=[],
    )
    assert code == "fsm_invalid"


def test_fsm_capability_proposal_is_built_before_validation():
    proposal = {
        "task": {"goal": "query device alarms", "inputs": {"device_id": "NE-1"}},
        "selected_capabilities": [
            {"agent_id": "noc-diagnosis-agent", "capability_id": "alarm.query"}
        ],
    }

    result = _validate_plan_candidate(
        proposal,
        fsm_mode=True,
        fsm_registry=list(default_agent_cards()),
        planner_mode="proposal",
        task_description="query device alarms",
    )
    assert result[0] is None
    assert result[2] is not None and result[2].valid


def test_plan_json_parser_accepts_fenced_json_and_error_has_bounded_attempts():
    parsed = _parse_plan_json('```json\n{"title":"x","description":"y","steps":[]}\n```')
    assert parsed["title"] == "x"

    event = _plan_error_event("plan_json_invalid", "invalid", attempts=2, model_calls=2)
    assert event["code"] == "plan_json_invalid"
    assert event["attempts"] == 2
    assert event["model_calls"] == 2


def test_plan_generation_retries_once_in_same_agent_session(monkeypatch):
    import core.services.ontology_service as ontology_service
    import core.services.project_scope as project_scope
    import orchestration.subagents.plan_mode as plan_mode

    valid_json = '{"title":"x","description":"y","steps":[{"title":"s","description":"d"}]}'
    agent_instances = []
    stream_histories = []

    class FakeStreamingAgent:
        def __init__(self, agent, mcp_clients):
            self.agent = agent
            self._call_index = len(agent_instances)
            agent_instances.append(self)

        async def stream(self, history, context):
            stream_histories.append(history)
            yield "text_delta", "not json" if self._call_index == 0 else valid_json

        def get_usage(self):
            return {"llm_call_count": 1}

    class FakePlanService:
        def __init__(self, db):
            pass

        def create_plan(self, **kwargs):
            return SimpleNamespace(plan_id="plan-1")

        def update_plan(self, *args, **kwargs):
            pass

        @staticmethod
        def plan_to_dict(plan):
            return {"plan_id": plan.plan_id, "title": "x", "description": "y", "steps": []}

    async def fake_create_agent_executor(**kwargs):
        return object(), []

    async def fake_close_clients(clients):
        pass

    async def fake_prepare_history(messages, model_name):
        return []

    monkeypatch.setattr(plan_mode, "_load_visible_agents", lambda db, user_id, enabled: [])
    monkeypatch.setattr(plan_mode, "_fsm_planning_enabled", lambda: False)
    monkeypatch.setattr(plan_mode, "_resolve_plugin_capabilities", lambda *args: ([], [], []))
    monkeypatch.setattr(plan_mode, "_build_tools_description", lambda *args: "")
    monkeypatch.setattr(plan_mode, "_load_plan_prompt", lambda tools: "PLAN")
    monkeypatch.setattr(plan_mode, "_build_file_context", lambda *args, **kwargs: "")
    monkeypatch.setattr(plan_mode, "_prepare_history", fake_prepare_history)
    monkeypatch.setattr(plan_mode, "create_agent_executor", fake_create_agent_executor)
    monkeypatch.setattr(plan_mode, "StreamingAgent", FakeStreamingAgent)
    monkeypatch.setattr(plan_mode, "close_clients", fake_close_clients)
    monkeypatch.setattr(plan_mode, "PlanService", FakePlanService)
    monkeypatch.setattr(project_scope, "build_project_ctx_from_chat_id", lambda db, chat_id: "")
    monkeypatch.setattr(
        ontology_service,
        "build_user_ontology_runtime",
        lambda **kwargs: (False, None),
    )

    async def collect():
        return [
            event
            async for event in astream_generate_plan("task", "user-1", object())
        ]

    events = asyncio.run(collect())
    assert [event["type"] for event in events] == [
        "plan_generating",
        "plan_retrying",
        "plan_generated",
    ]
    assert len(agent_instances) == 2
    assert agent_instances[0].agent is agent_instances[1].agent
    assert stream_histories[0][-1]["content"].endswith("用户任务：task")
    assert stream_histories[1][:-1] == stream_histories[0]
    assert stream_histories[1][-1] == {"role": "user", "content": plan_mode._JSON_RETRY_PROMPT}


def test_plan_generation_stops_after_one_retry(monkeypatch):
    import core.services.ontology_service as ontology_service
    import core.services.project_scope as project_scope
    import orchestration.subagents.plan_mode as plan_mode

    class AlwaysInvalidStreamingAgent:
        def __init__(self, agent, mcp_clients):
            pass

        async def stream(self, history, context):
            yield "text_delta", "not json"

        def get_usage(self):
            return {"llm_call_count": 1}

    async def fake_create_agent_executor(**kwargs):
        return object(), []

    async def fake_close_clients(clients):
        pass

    async def fake_prepare_history(messages, model_name):
        return []

    monkeypatch.setattr(plan_mode, "_load_visible_agents", lambda db, user_id, enabled: [])
    monkeypatch.setattr(plan_mode, "_fsm_planning_enabled", lambda: False)
    monkeypatch.setattr(plan_mode, "_resolve_plugin_capabilities", lambda *args: ([], [], []))
    monkeypatch.setattr(plan_mode, "_build_tools_description", lambda *args: "")
    monkeypatch.setattr(plan_mode, "_load_plan_prompt", lambda tools: "PLAN")
    monkeypatch.setattr(plan_mode, "_build_file_context", lambda *args, **kwargs: "")
    monkeypatch.setattr(plan_mode, "_prepare_history", fake_prepare_history)
    monkeypatch.setattr(plan_mode, "create_agent_executor", fake_create_agent_executor)
    monkeypatch.setattr(plan_mode, "StreamingAgent", AlwaysInvalidStreamingAgent)
    monkeypatch.setattr(plan_mode, "close_clients", fake_close_clients)
    monkeypatch.setattr(project_scope, "build_project_ctx_from_chat_id", lambda db, chat_id: "")
    monkeypatch.setattr(
        ontology_service,
        "build_user_ontology_runtime",
        lambda **kwargs: (False, None),
    )

    async def collect():
        return [
            event
            async for event in astream_generate_plan("task", "user-1", object())
        ]

    events = asyncio.run(collect())
    error = events[-1]
    assert error["type"] == "plan_error"
    assert error["code"] == "plan_json_invalid"
    assert error["attempts"] == 2
    assert error["model_calls"] == 2
    assert sum(event["type"] == "plan_retrying" for event in events) == 1


def test_plan_step_ids_are_namespaced_by_plan():
    class FakeDB:
        def __init__(self):
            self.added = []

        def add(self, value):
            self.added.append(value)

        def commit(self):
            pass

        def refresh(self, value):
            pass

    db = FakeDB()
    service = PlanService(db)
    service.create_plan(
        user_id="user-1",
        title="one",
        task_input="task",
        steps=[{"step_id": "s1", "title": "s1"}],
    )
    service.create_plan(
        user_id="user-1",
        title="two",
        task_input="task",
        steps=[{"step_id": "s1", "title": "s1"}],
    )

    step_ids = [item.step_id for item in db.added if item.__class__.__name__ == "PlanStep"]
    assert len(step_ids) == len(set(step_ids)) == 2
    assert all(item.startswith("plan_") and "_step_1" in item for item in step_ids)


def test_safeharness_block_event_keeps_judge_evidence_and_skips_planner(monkeypatch):
    import orchestration.subagents.plan_mode as plan_mode

    async def fake_guard(*args, **kwargs):
        return (
            "semantic_prompt_injection_uncertain",
            "task",
            [],
            {
                "judge_status": "ok",
                "judge_calls": 1,
                "judge_model": "judge-model",
                "semantic_label": "UNCERTAIN",
                "semantic_confidence": 0.2,
                "judge_result_class": "low_confidence_safe",
            },
        )

    monkeypatch.setattr(plan_mode, "_fsm_planning_enabled", lambda: True)
    monkeypatch.setattr(plan_mode, "_load_visible_agents", lambda *args: [])
    monkeypatch.setattr(plan_mode, "_safeharness_check_fsm_inputs", fake_guard)

    async def forbidden_model_call(**kwargs):
        raise AssertionError("planner model must not be called")

    monkeypatch.setattr(plan_mode, "create_agent_executor", forbidden_model_call)

    async def collect():
        return [event async for event in astream_generate_plan("task", "user-1", object())]

    events = asyncio.run(collect())
    error = events[-1]
    assert error["type"] == "plan_error"
    assert error["code"] == "semantic_prompt_injection_uncertain"
    assert error["model_calls"] == 0
    assert error["judge_calls"] == 1
    assert error["judge_model"] == "judge-model"
    assert error["semantic_label"] == "UNCERTAIN"
    assert error["semantic_confidence"] == 0.2
    assert error["judge_result_class"] == "low_confidence_safe"
    assert not any(event["type"] == "plan_retrying" for event in events)


def test_safeharness_unavailable_event_is_actionable_and_skips_planner(monkeypatch):
    import orchestration.subagents.plan_mode as plan_mode

    async def fake_guard(*args, **kwargs):
        return (
            "safeharness_judge_unavailable",
            "task",
            [],
            {
                "judge_status": "unavailable",
                "judge_calls": 0,
                "judge_model": "",
                "semantic_label": "",
                "semantic_confidence": None,
                "judge_result_class": "unavailable",
            },
        )

    monkeypatch.setattr(plan_mode, "_fsm_planning_enabled", lambda: True)
    monkeypatch.setattr(plan_mode, "_load_visible_agents", lambda *args: [])
    monkeypatch.setattr(plan_mode, "_safeharness_check_fsm_inputs", fake_guard)

    async def forbidden_model_call(**kwargs):
        raise AssertionError("planner model must not be called")

    monkeypatch.setattr(plan_mode, "create_agent_executor", forbidden_model_call)

    async def collect():
        return [event async for event in astream_generate_plan("task", "user-1", object())]

    error = asyncio.run(collect())[-1]
    assert error["code"] == "safeharness_judge_unavailable"
    assert error["model_calls"] == 0
    assert error["judge_calls"] == 0
    assert error["judge_result_class"] == "unavailable"


def test_actual_safeharness_protocol_error_is_classified_before_planner(monkeypatch):
    import orchestration.subagents.plan_mode as plan_mode

    class FakeJudge:
        model = "judge-model"

        async def judge(self, request):
            return JudgeResult(
                "UNCERTAIN",
                0.0,
                "protocol failure",
                "test",
                self.model,
                "protocol_error",
                1,
                "protocol_error",
            )

    monkeypatch.setattr(plan_mode, "_fsm_planning_enabled", lambda: True)
    monkeypatch.setattr(plan_mode, "_load_visible_agents", lambda *args: [])
    monkeypatch.setattr(plan_mode, "_resolve_fsm_judge", lambda provider_id: FakeJudge())

    async def forbidden_model_call(**kwargs):
        raise AssertionError("planner model must not be called")

    monkeypatch.setattr(plan_mode, "create_agent_executor", forbidden_model_call)

    async def collect():
        return [event async for event in astream_generate_plan("查询当前告警", "user-1", object())]

    error = asyncio.run(collect())[-1]
    assert error["code"] == "safeharness_judge_protocol_error"
    assert error["model_calls"] == 0
    assert error["judge_calls"] == 1
    assert error["judge_model"] == "judge-model"
    assert error["semantic_label"] == "UNCERTAIN"
    assert error["semantic_confidence"] == 0.0
    assert error["judge_result_class"] == "protocol_error"
