"""Public build-edition and SafeHarness capability probes."""

from typing import Optional

from api.middleware.edition import edition_probe_payload
from api.routes.v1.safeharness import (
    capability_discovery_payload,
    safeharness_error_response,
    validate_discovery_query,
)
from core.auth.backend import UserContext, require_auth
from core.db.engine import get_db
from core.infra.responses import success_response
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

router = APIRouter(prefix="/v1/meta", tags=["meta"])


@router.get("/edition", summary="当前部署的版本与能力位")
async def get_edition():
    return success_response(data=edition_probe_payload())


@router.get("/safeharness", summary="SafeHarness capability discovery")
async def get_safeharness_capabilities(
    request: Request,
    conversation_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    project_id: Optional[str] = None,
    user: Optional[UserContext] = Depends(require_auth(False)),
    db: Session = Depends(get_db),
):
    """Return a fail-closed SafeHarness capability contract.

    Availability requires authenticated authority context plus a successful
    live Admin UI probe. Credentials and raw upstream state are never returned.
    """
    try:
        validate_discovery_query(request)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return safeharness_error_response(
            exc.status_code,
            str(detail.get("code") or "safeharness_invalid_query"),
            str(detail.get("message") or "SafeHarness discovery query is invalid."),
        )
    data = await capability_discovery_payload(
        user,
        db,
        conversation_id=conversation_id,
        agent_id=agent_id,
        project_id=project_id,
    )
    return success_response(data=data, trace_id=data["context"]["trace_id"])
