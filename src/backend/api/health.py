"""Health-check, readiness, and liveness endpoints."""

import os
import sqlite3
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.schemas import HealthResponse
from core.config.settings import settings
from core.db.engine import get_db
from core.infra.logging import get_logger
from core.storage import get_storage

logger = get_logger(__name__)

router = APIRouter(tags=["monitoring"])


def _safeharness_health_detail() -> bool:
    return (os.environ.get("SAFEHARNESS_HEALTH_DETAIL") or "").strip().lower() in ("1", "true", "yes")


async def _safeharness_component_status() -> dict:
    """Optional /ready addendum (P3.1, gated by SAFEHARNESS_HEALTH_DETAIL).

    Informational only: these results never flip ``ready`` — the platform
    readiness verdict stays the pre-P3.1 computation. No secret values and no
    filesystem paths are reported, only set/unset and ok/not-ok.
    """
    from core.safeharness_audit import resolve_audit_db_path

    status: dict = {"protected_chat_enabled": _env_truthy(os.environ.get("SAFEHARNESS_PROTECTED_CHAT_ENABLED"))}

    audit_path = resolve_audit_db_path()
    audit_info = {"path_set": bool(audit_path), "ok": False}
    if audit_path:
        try:
            with sqlite3.connect(f"file:{audit_path}?mode=ro", uri=True) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('safeharness_audit','audit_records')"
                ).fetchone()
            audit_info["ok"] = bool(row and row[0])
        except sqlite3.Error as e:
            logger.warning("safeharness_audit_db_check_failed", error=str(e))
    status["audit_db"] = audit_info

    gateway_url = (os.environ.get("SAFEHARNESS_TOOL_GATEWAY_URL") or "").strip()
    gateway_key = (os.environ.get("SAFEHARNESS_TOOL_GATEWAY_API_KEY") or "").strip()
    gateway_info = {"url_configured": bool(gateway_url and gateway_key), "ok": None}
    if gateway_url and gateway_key:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(
                    gateway_url.rstrip("/") + "/health",
                    headers={"Authorization": f"Bearer {gateway_key}"},
                )
            gateway_info["ok"] = resp.status_code == 200
        except Exception as e:
            logger.warning("safeharness_tool_gateway_probe_failed", error=str(e))
            gateway_info["ok"] = False
    status["tool_gateway"] = gateway_info
    return status


def _env_truthy(value) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes")


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """
    基础健康检查

    用于监控和负载均衡器的健康检查端点。
    仅检查服务是否响应，不检查依赖。
    """
    return HealthResponse(
        status="healthy",
        service="sumc",
        timestamp=datetime.now().isoformat(),
    )


@router.get("/ready", summary="就绪探针")
async def readiness_check(db: Session = Depends(get_db)):
    """
    就绪检查（包含依赖检查）

    检查服务及其依赖是否就绪。
    用于 Kubernetes readiness probe。
    """
    checks = {}

    # Check database
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))
        checks["database"] = False

    # Check storage
    try:
        storage = get_storage()
        checks["storage"] = storage is not None
    except Exception as e:
        logger.error("storage_health_check_failed", error=str(e))
        checks["storage"] = False

    # Check Redis
    redis_url = settings.redis.url
    if redis_url:
        try:
            from core.infra.redis import get_redis
            r = get_redis()
            await r.ping()
            checks["redis"] = True
        except Exception as e:
            logger.error("redis_health_check_failed", error=str(e))
            checks["redis"] = False

    # Check user center
    auth_mode = settings.auth.mode
    if auth_mode != "mock":
        try:
            checks["user_center"] = True
        except Exception as e:
            logger.error("user_center_health_check_failed", error=str(e))
            checks["user_center"] = False
    else:
        checks["user_center"] = True

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503

    content = {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }
    if _safeharness_health_detail():
        content["safeharness"] = await _safeharness_component_status()

    return JSONResponse(
        status_code=status_code,
        content=content,
    )


@router.get("/live", summary="存活探针")
async def liveness_check():
    """
    存活检查（简单响应）

    最简单的存活检查，仅返回服务存活状态。
    用于 Kubernetes liveness probe。
    """
    return {"alive": True, "timestamp": datetime.now().isoformat()}
