import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import sessionmaker

from api.routes.v1 import plans as plan_routes
from core.auth.backend import UserContext
from core.db.models import Plan, UserShadow


def test_persisted_approval_resumes_once_after_session_restart(db_session, monkeypatch):
    plan_hash = "a" * 64
    approval = {
        "approval_id": "approval-1",
        "state_id": "approve_restart",
        "approval_for": "restart_interface",
        "run_id": "run-before-restart",
        "plan_hash": plan_hash,
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
        "used": False,
    }
    db_session.add(UserShadow(user_id="user-1", username="operator"))
    db_session.add(
        Plan(
            plan_id="plan-1",
            user_id="user-1",
            title="Recover approval",
            task_input="repair device",
            status="running",
            extra_data={
                "chat_id": "chat-1",
                "fsm_runtime": {
                    "status": "awaiting_approval",
                    "current_state_id": "approve_restart",
                    "completed_state_ids": ["query_before", "diagnose_before"],
                    "approval": approval,
                    "checkpoint_version": 3,
                },
            },
        )
    )
    db_session.commit()

    # A new SQLAlchemy session models process/session restart: no in-memory
    # controller or approval object is reused.
    restarted_db = sessionmaker(bind=db_session.get_bind())()
    calls = []

    async def start_plan_execute_run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_id="run-after-restart")

    monkeypatch.setattr(plan_routes, "_load_chat_history", lambda *_args: [])
    monkeypatch.setattr(plan_routes, "resolve_effective_chat_model_name", lambda: "test-model")
    monkeypatch.setattr(
        "orchestration.chat_run_executor.start_plan_execute_run",
        start_plan_execute_run,
    )
    request = plan_routes.PlanApprovalDecisionRequest(
        decision="approved",
        run_id=approval["run_id"],
        state_id=approval["state_id"],
        plan_hash=plan_hash,
    )
    user = UserContext(user_id="user-1", user_center_id="center-1", username="operator")

    response = asyncio.run(
        plan_routes.decide_plan_approval(
            "plan-1", approval["approval_id"], request, user, restarted_db
        )
    )
    assert response["data"] == {
        "approval_id": "approval-1",
        "decision": "approved",
        "expired": False,
        "run_id": "run-after-restart",
    }
    assert calls[0]["approval_decisions"] == {"approve_restart": "approved"}

    restarted_db.expire_all()
    saved = restarted_db.query(Plan).filter(Plan.plan_id == "plan-1").one()
    saved_approval = saved.extra_data["fsm_runtime"]["approval"]
    assert saved_approval["used"] is True
    assert saved_approval["decision"] == "approved"

    with pytest.raises(HTTPException) as repeated:
        asyncio.run(
            plan_routes.decide_plan_approval(
                "plan-1", approval["approval_id"], request, user, restarted_db
            )
        )
    assert repeated.value.status_code == 409
    assert len(calls) == 1
    restarted_db.close()
