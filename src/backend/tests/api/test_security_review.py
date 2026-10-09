"""Review workbench API: queue assembly, evidence pack, and resolution recording.

The review state lives in plans.extra_data.fsm_runtime.status (the plans.status
CHECK constraint has no 'manual_review' value — the row reads 'failed'), so the
queue must select on the JSON payload, not the status column.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from core.auth.backend import UserContext, get_current_user
from core.db.engine import Base, get_db
from core.db.models import Plan, UserShadow


@pytest.fixture
def client():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    seed = Session()
    seed.add(UserShadow(user_id="u1", username="alice"))
    seed.commit()
    seed.add_all(
        [
            Plan(
                plan_id="p_review",
                user_id="u1",
                title="重启网元 NE-WH-OLT-01",
                task_input="重启网元 NE-WH-OLT-01",
                status="failed",
                total_steps=3,
                completed_steps=1,
                result_summary="Manual review required",
                extra_data={
                    "fsm_runtime": {
                        "status": "manual_review",
                        "completed_states": ["s1"],
                        "error_class": None,
                    }
                },
            ),
            Plan(
                plan_id="p_plain_failure",
                user_id="u1",
                title="普通失败计划",
                task_input="task",
                status="failed",
                total_steps=2,
                completed_steps=0,
                result_summary="model error",
                extra_data={},
            ),
            Plan(
                plan_id="p_resolved",
                user_id="u1",
                title="已结案计划",
                task_input="task",
                status="completed",
                total_steps=3,
                completed_steps=2,
                result_summary="Reviewed: no violation by alice",
                extra_data={
                    "review": {
                        "resolution": "close_acknowledged",
                        "note": "ok",
                        "reviewer": "alice",
                        "resolved_at": "2026-09-21T10:00:00",
                        "duration_ms": 300000,
                    }
                },
            ),
            Plan(
                plan_id="p_running",
                user_id="u1",
                title="运行中计划",
                task_input="task",
                status="running",
                total_steps=1,
                completed_steps=0,
                extra_data={},
            ),
        ]
    )
    seed.commit()
    seed.close()

    from api.app import app

    def _override_db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="u1", user_center_id="c", username="alice",
    )
    app.dependency_overrides[get_db] = _override_db
    c = TestClient(app)
    try:
        yield c
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)


def test_queue_lists_only_review_items_with_evidence(client):
    res = client.get("/v1/security/review/queue")
    assert res.status_code == 200
    data = res.json()["data"]
    ids = [item["plan_id"] for item in data["items"]]
    assert ids == ["p_review"]

    item = data["items"][0]
    assert item["fsm_runtime"]["status"] == "manual_review"
    assert item["fsm_runtime"]["completed_states"] == ["s1"]
    assert item["waiting_ms"] is not None
    assert item["review"] is None
    assert "p_plain_failure" not in ids and "p_running" not in ids

    assert data["stats"]["resolved_count"] == 1
    assert data["stats"]["avg_duration_ms"] == 300000
    assert data["resolved"][0]["plan_id"] == "p_resolved"


def test_resolve_acknowledged_records_review_and_closes(client):
    res = client.post(
        "/v1/security/review/p_review/resolve",
        json={"resolution": "close_acknowledged", "note": "benign drill"},
    )
    assert res.status_code == 200
    item = res.json()["data"]
    assert item["status"] == "completed"
    assert item["review"]["resolution"] == "close_acknowledged"
    assert item["review"]["reviewer"] == "alice"
    assert item["review"]["duration_ms"] is not None

    # the resolved row moved out of the queue into the history list
    data = client.get("/v1/security/review/queue").json()["data"]
    assert all(i["plan_id"] != "p_review" for i in data["items"])
    assert any(r["plan_id"] == "p_review" for r in data["resolved"])

    # resolving twice is rejected
    again = client.post(
        "/v1/security/review/p_review/resolve",
        json={"resolution": "close_rejected"},
    )
    assert again.status_code == 409


def test_resolve_rejected_marks_plan_failed(client):
    res = client.post(
        "/v1/security/review/p_review/resolve",
        json={"resolution": "close_rejected", "note": "real violation"},
    )
    assert res.status_code == 200
    assert res.json()["data"]["status"] == "failed"


def test_resolve_validation_errors(client):
    assert (
        client.post(
            "/v1/security/review/unknown/resolve",
            json={"resolution": "close_acknowledged"},
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/v1/security/review/p_plain_failure/resolve",
            json={"resolution": "close_acknowledged"},
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/v1/security/review/p_review/resolve",
            json={"resolution": "not_a_resolution"},
        ).status_code
        == 400
    )

def _seed_supervision_audit(tmp_path):
    """Create an audit ledger with one supervision deny row."""
    import os
    import sqlite3

    db_path = tmp_path / "sup-audit.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE safeharness_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, record_type TEXT, "
        "event_kind TEXT, tenant_id TEXT, session_id TEXT, run_id TEXT, trace_id TEXT, agent_id TEXT, "
        "framework TEXT, decision_disposition TEXT, decision_reason_code TEXT, event_payload TEXT, "
        "details TEXT, occurred_at TEXT, written_at TEXT)"
    )
    payload = (
        "{'source': 'external_agent_supervision', 'framework': 'smolagents', 'trigger': 'max_steps', "
        "'intervention': {'trigger': 'max_steps', 'detail': '步数超过上限 8', 'step': 3}, "
        "'review': 'ok:content_provenance_attached', 'redacted': True}"
    )
    conn.execute(
        "INSERT INTO safeharness_audit (record_type, event_kind, tenant_id, session_id, run_id, trace_id, "
        "agent_id, framework, decision_disposition, decision_reason_code, event_payload, occurred_at, written_at) "
        "VALUES ('decision', 'content_received', 't', 's', 'run-x', 'tr', 'ext-smolagents', 'sumc-external-agents', "
        "'deny', 'supervision_max_steps', ?, '2026-09-30T10:00:00', '2026-09-30T10:00:01')",
        (payload,),
    )
    conn.commit()
    conn.close()
    return db_path


def test_queue_includes_supervision_interventions(client, tmp_path, monkeypatch):
    db_path = _seed_supervision_audit(tmp_path)
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", str(db_path))
    body = client.get("/v1/security/review/queue").json()["data"]
    sup = [i for i in body["items"] if i.get("kind") == "supervision"]
    assert len(sup) == 1
    item = sup[0]
    assert item["plan_id"] == "supervision-1"
    assert item["status"] == "manual_review"
    assert item["fsm_runtime"]["current_state"] == "max_steps"
    assert item["evidence"]["framework"] == "smolagents"
    assert item["evidence"]["semantic_review"] == "ok:content_provenance_attached"


def test_supervision_resolve_moves_to_resolved(client, tmp_path, monkeypatch):
    db_path = _seed_supervision_audit(tmp_path)
    monkeypatch.setenv("SAFEHARNESS_AUDIT_DB", str(db_path))

    # Unknown id → 404 (not present in the ledger).
    r = client.post(
        "/v1/security/review/supervision-999/resolve",
        json={"resolution": "close_acknowledged", "note": ""},
    )
    assert r.status_code == 404

    r = client.post(
        "/v1/security/review/supervision-1/resolve",
        json={"resolution": "close_rejected", "note": "确认误触发，已调阈值"},
    )
    assert r.status_code == 200
    assert r.json()["data"]["kind"] == "supervision"

    body = client.get("/v1/security/review/queue").json()["data"]
    assert not [i for i in body["items"] if i.get("kind") == "supervision"]
    sup_resolved = [x for x in body["resolved"] if x["plan_id"] == "supervision-1"]
    assert len(sup_resolved) == 1
    assert sup_resolved[0]["resolution"] == "close_rejected"
    assert sup_resolved[0]["note"] == "确认误触发，已调阈值"
