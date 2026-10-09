"""Operator review workbench for FSM plans halted in manual_review.

Every rejected or expired plan approval lands the FSM run in a
``manual_review`` terminal state. The workbench turns those rows into an
evidence-backed queue: reviewers see the plan, its execution trace, and its
review history in one payload, and resolutions are recorded so review
duration (open → resolved) becomes measurable.

Demo scope: any authenticated user may review (the offline demo has a single
operator account); a production deployment gates this router on a security
role.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.auth.backend import get_current_user, UserContext
from core.db.engine import get_db
from core.db.models import Plan
from core.infra.responses import success_response
from core.services.plan_service import PlanService

router = APIRouter(prefix="/v1/security/review", tags=["SecurityReview"])

_REVIEW_RESOLUTIONS = {
    # resolution -> (plan status, result summary prefix)
    "close_acknowledged": ("completed", "Reviewed: no violation"),
    "close_rejected": ("failed", "Reviewed: violation confirmed"),
}

# Review-workbench id prefix for external-agent supervision interventions.
# These live in the SafeHarness audit ledger, not the plans table; dispositions
# are stored in system_configs under this key.
_SUPERVISION_PREFIX = "supervision-"
_SUPERVISION_REVIEWS_KEY = "external_agent_supervision_reviews"


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _supervision_rows() -> List[Dict[str, Any]]:
    """Read supervision deny rows from the SafeHarness audit ledger (read-only).

    The audit sink stores event payloads as their python dict repr, so the
    mappingproxy entries that leak into them are normalised before parsing.
    """
    import ast
    import json
    import os
    import sqlite3

    try:
        from core.safeharness_audit import resolve_audit_db_path

        db_path = resolve_audit_db_path()
        if not db_path or not os.path.exists(db_path):
            return []
        rows = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True).execute(
            "SELECT id, occurred_at, agent_id, framework, run_id, decision_reason_code, event_payload "
            "FROM safeharness_audit "
            "WHERE record_type='decision' AND decision_disposition='deny' "
            "AND decision_reason_code LIKE 'supervision_%' "
            "ORDER BY id DESC LIMIT 200"
        ).fetchall()
    except Exception:  # noqa: BLE001 — ledger absence keeps the queue usable
        return []

    items: List[Dict[str, Any]] = []
    for row in rows:
        audit_id, occurred_at, agent_id, framework, run_id, reason_code, payload_raw = row
        payload: Dict[str, Any] = {}
        try:
            payload = ast.literal_eval((payload_raw or "").replace("mappingproxy(", "("))
            if not isinstance(payload, dict):
                payload = {}
        except Exception:  # noqa: BLE001 — a broken payload degrades to empty
            payload = {}
        items.append(
            {
                "audit_id": audit_id,
                "occurred_at": occurred_at,
                "agent_id": agent_id or "",
                "framework": framework or "",
                "run_id": run_id or "",
                "reason_code": reason_code or "",
                "payload": payload,
            }
        )
    return items


def _supervision_reviews(db: Session) -> Dict[str, Any]:
    import json

    from core.db.models import SystemConfig

    try:
        row = db.get(SystemConfig, _SUPERVISION_REVIEWS_KEY)
        if row is None or not (row.config_value or "").strip():
            return {}
        raw = json.loads(row.config_value)
        return raw if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _save_supervision_reviews(db: Session, reviews: Dict[str, Any]) -> None:
    import json

    from core.db.models import SystemConfig

    row = db.get(SystemConfig, _SUPERVISION_REVIEWS_KEY)
    payload = json.dumps(reviews, ensure_ascii=False)
    if row is None:
        db.add(
            SystemConfig(
                config_key=_SUPERVISION_REVIEWS_KEY,
                config_value=payload,
                display_name="外部智能体监督复核结论",
                description="supervision_* 审计行的人工复核处置记录",
                group_key="external_agents",
                is_secret=False,
            )
        )
    else:
        row.config_value = payload
    db.commit()


def _supervision_item(row: Dict[str, Any], now: datetime, review: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = row.get("payload") or {}
    intervention = payload.get("intervention") if isinstance(payload.get("intervention"), dict) else {}
    trigger = str(intervention.get("trigger") or payload.get("trigger") or row.get("reason_code") or "supervision")
    opened = _parse_iso(row.get("occurred_at"))
    if opened is not None and opened.tzinfo is not None:
        opened = opened.replace(tzinfo=None)
    item_id = f"{_SUPERVISION_PREFIX}{row['audit_id']}"
    item: Dict[str, Any] = {
        "plan_id": item_id,
        "kind": "supervision",
        "user_id": row.get("agent_id"),
        "title": f"运行监督干预：{trigger}",
        "task_input": f"外部智能体 {payload.get('framework') or row.get('framework')}（{row.get('agent_id')}）运行被监督截断（run {row.get('run_id')}）。",
        "status": "manual_review",
        "result_summary": str(intervention.get("detail") or trigger),
        "total_steps": intervention.get("step"),
        "completed_steps": None,
        "created_at": row.get("occurred_at"),
        "updated_at": row.get("occurred_at"),
        "waiting_ms": max(0, int((now - opened).total_seconds() * 1000)) if opened else None,
        "review": review,
        "steps": [],
    }
    item["fsm_runtime"] = {
        "status": "review_closed" if review else "manual_review",
        "current_state": trigger,
        "error_class": row.get("reason_code"),
        "result_summary": str(intervention.get("detail") or trigger),
    }
    item["evidence"] = {
        "framework": payload.get("framework") or row.get("framework"),
        "agent_id": row.get("agent_id"),
        "run_id": row.get("run_id"),
        "trigger": trigger,
        "intervention": intervention,
        "semantic_review": payload.get("review"),
    }
    return item


def _review_item(plan: Plan, now: datetime) -> Dict[str, Any]:
    payload = PlanService.plan_to_dict(plan)
    extra = plan.extra_data if isinstance(plan.extra_data, dict) else {}
    review = extra.get("review") if isinstance(extra.get("review"), dict) else None
    opened = _parse_iso(payload.get("updated_at")) or _parse_iso(payload.get("created_at"))
    item: Dict[str, Any] = {
        "plan_id": plan.plan_id,
        "user_id": plan.user_id,
        "title": plan.title,
        "task_input": plan.task_input,
        "status": plan.status,
        "result_summary": plan.result_summary,
        "total_steps": plan.total_steps,
        "completed_steps": plan.completed_steps,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "waiting_ms": max(0, int((now - opened).total_seconds() * 1000)) if opened else None,
        "review": review,
    }
    fsm_runtime = extra.get("fsm_runtime") if isinstance(extra.get("fsm_runtime"), dict) else {}
    item["fsm_runtime"] = {
        key: fsm_runtime.get(key)
        for key in (
            "status",
            "completed_states",
            "pending_approval",
            "current_state",
            "error_class",
            "result_summary",
        )
        if key in fsm_runtime
    }
    item["steps"] = [
        {
            "step_id": step.get("step_id"),
            "title": step.get("title"),
            "status": step.get("status"),
            "error_message": step.get("error_message"),
            "started_at": step.get("started_at"),
            "completed_at": step.get("completed_at"),
        }
        for step in payload.get("steps", [])
    ]
    return item


class ResolveReviewRequest(BaseModel):
    resolution: str
    note: str = ""


def _fsm_runtime(plan: Plan) -> Dict[str, Any]:
    extra = plan.extra_data if isinstance(plan.extra_data, dict) else {}
    runtime = extra.get("fsm_runtime")
    return runtime if isinstance(runtime, dict) else {}


def _in_review(plan: Plan) -> bool:
    """The review state lives in extra_data.fsm_runtime.status — plans.status is
    constrained to a fixed enum without 'manual_review' (the row reads 'failed')."""
    return _fsm_runtime(plan).get("status") == "manual_review"


@router.get("/queue", summary="人工复核队列（manual_review 计划 + 证据包）")
async def list_review_queue(
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List manual_review plans with evidence, plus resolved-duration stats."""
    candidates = (
        db.query(Plan)
        .filter(Plan.status.in_(["failed", "completed"]))
        .order_by(Plan.updated_at.desc())
        .limit(200)
        .all()
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    items: List[Dict[str, Any]] = []
    resolved: List[Dict[str, Any]] = []
    for plan in candidates:
        extra = plan.extra_data if isinstance(plan.extra_data, dict) else {}
        review = extra.get("review") if isinstance(extra.get("review"), dict) else None
        if review:
            resolved.append(
                {
                    "plan_id": plan.plan_id,
                    "title": plan.title,
                    "resolution": review.get("resolution"),
                    "note": review.get("note", ""),
                    "reviewer": review.get("reviewer"),
                    "resolved_at": review.get("resolved_at"),
                    "duration_ms": review.get("duration_ms"),
                }
            )
            continue
        if _in_review(plan):
            items.append(_review_item(plan, now))

    # External-agent supervision interventions join the same queue: open ones
    # become items, dispositions are recorded in system_configs.
    reviews = _supervision_reviews(db)
    for row in _supervision_rows():
        item_id = f"{_SUPERVISION_PREFIX}{row['audit_id']}"
        review = reviews.get(item_id)
        if isinstance(review, dict):
            resolved.append(
                {
                    "plan_id": item_id,
                    "title": f"运行监督干预：{(row.get('payload') or {}).get('trigger', row.get('reason_code'))}",
                    "resolution": review.get("resolution"),
                    "note": review.get("note", ""),
                    "reviewer": review.get("reviewer"),
                    "resolved_at": review.get("resolved_at"),
                    "duration_ms": review.get("duration_ms"),
                }
            )
            continue
        items.append(_supervision_item(row, now, None))

    durations = [r["duration_ms"] for r in resolved if isinstance(r.get("duration_ms"), (int, float))]
    stats = {
        "resolved_count": len(resolved),
        "avg_duration_ms": int(sum(durations) / len(durations)) if durations else None,
        "max_duration_ms": int(max(durations)) if durations else None,
    }
    return success_response(data={"items": items, "resolved": resolved[:20], "stats": stats})


@router.post("/{plan_id}/resolve", summary="登记人工复核结论")
async def resolve_review_item(
    plan_id: str,
    req: ResolveReviewRequest,
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record the reviewer disposition and close the review item."""
    if req.resolution not in _REVIEW_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"resolution 只能为: {', '.join(sorted(_REVIEW_RESOLUTIONS))}",
        )

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Supervision interventions close into system_configs (the audit ledger is
    # immutable); the queue reads the disposition map back.
    if plan_id.startswith(_SUPERVISION_PREFIX):
        reviews = _supervision_reviews(db)
        if plan_id not in reviews:
            existing = {f"{_SUPERVISION_PREFIX}{row['audit_id']}" for row in _supervision_rows()}
            if plan_id not in existing:
                raise HTTPException(status_code=404, detail="监督复核条目不存在")
        opened = None
        for row in _supervision_rows():
            if f"{_SUPERVISION_PREFIX}{row['audit_id']}" == plan_id:
                opened = _parse_iso(row.get("occurred_at"))
                break
        if opened is not None and opened.tzinfo is not None:
            opened = opened.replace(tzinfo=None)
        duration_ms = (
            max(0, int((now - opened).total_seconds() * 1000)) if opened else None
        )
        reviews = {
            **reviews,
            plan_id: {
                "resolution": req.resolution,
                "note": req.note,
                "reviewer": user.username,
                "resolved_at": now.isoformat(),
                "duration_ms": duration_ms,
            },
        }
        _save_supervision_reviews(db, reviews)
        return success_response(
            data={"plan_id": plan_id, "kind": "supervision", "review": reviews[plan_id]}
        )

    plan = db.query(Plan).filter(Plan.plan_id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="计划不存在")
    if not _in_review(plan):
        raise HTTPException(status_code=409, detail="计划不在人工复核状态")

    opened = plan.updated_at or plan.created_at
    if opened and opened.tzinfo is None:
        duration_ms = max(0, int((now - opened).total_seconds() * 1000))
    else:
        duration_ms = None
    status, summary_prefix = _REVIEW_RESOLUTIONS[req.resolution]
    extra = plan.extra_data if isinstance(plan.extra_data, dict) else {}
    fsm_runtime = extra.get("fsm_runtime") if isinstance(extra.get("fsm_runtime"), dict) else {}
    fsm_runtime = {**fsm_runtime, "status": "review_closed"}
    extra = {
        **extra,
        "fsm_runtime": fsm_runtime,
        "review": {
            "resolution": req.resolution,
            "note": req.note,
            "reviewer": user.username,
            "resolved_at": now.isoformat(),
            "duration_ms": duration_ms,
            "prior_status": plan.status,
        },
    }
    svc = PlanService(db)
    svc.update_plan(
        plan_id,
        status=status,
        result_summary=f"{summary_prefix} by {user.username}"
        + (f": {req.note}" if req.note else ""),
        extra_data=extra,
    )
    plan = svc.get_plan(plan_id, user.user_id) or db.query(Plan).filter(Plan.plan_id == plan_id).first()
    return success_response(data=_review_item(plan, now))
