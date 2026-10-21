"""Personal system settings API (/v1/me/system) —— the service-config surface delegated in CE.

The CE derived tree has no Config system console (`service_configs.py` is EE, physically absent),
so deployment-level service configs such as the search-engine key have no UI to write in CE. This
route delegates a whitelist of groups **related to personal use** (internet search / file parsing /
knowledge base / sandbox / context) to the instance admin (gate = ``require_system_settings``:
CONFIG_TOKEN / can_system_config capability bit / CE+mock single trust domain). It is equally usable
under EE, but the frontend only shows the entry in CE (EE uses the Config console, to avoid a double entry).

Enterprise-governance groups (dingtalk / lark / auth / industry, etc.) are **not** in the whitelist,
and remain exclusive to the EE system console.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional

from api.deps import require_system_settings, user_can_manage_system_settings
from core.auth.backend import UserContext, require_auth
from core.config.settings import settings
from core.db.engine import get_db
from core.infra.responses import success_response
from core.services.service_probes import (
    config_row_to_dict,
    reinitialize_mcp_pool,
    test_service_group,
)
from core.services.system_config import SystemConfigService
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

router = APIRouter(prefix="/v1/me/system", tags=["My System Settings"])
logger = logging.getLogger(__name__)

# Whitelist of service-config groups related to personal use (key → display name).
# Enterprise workbench (dingtalk/lark), login session (auth), and EE industry services (industry) are not delegated.
PERSONAL_GROUPS: Dict[str, str] = {
    "internet_search": "互联网搜索",
    "file_parser": "文件解析服务",
    "knowledge_base": "知识库服务",
    "sandbox": "沙箱 / 代码执行",
    "context": "对话上下文压缩",
}

# Whitelist groups that support connectivity testing (sandbox/context are pure switches, no external service to test)
_TESTABLE_GROUPS = {"internet_search", "file_parser", "knowledge_base"}


def _personal_groups() -> Dict[str, str]:
    """Return the edition-specific settings surface.

    CE provides only owner-isolated local knowledge bases, so the external
    knowledge-base connector is not configurable there.
    """
    if settings.edition.edition == "ce":
        return {key: label for key, label in PERSONAL_GROUPS.items() if key != "knowledge_base"}
    return PERSONAL_GROUPS


class ConfigUpdateItem(BaseModel):
    key: str
    value: Optional[str] = None


class BulkUpdateRequest(BaseModel):
    items: List[ConfigUpdateItem]


def _svc() -> SystemConfigService:
    return SystemConfigService.get_instance()


# ── Access probe ─────────────────────────────────────────────────────────────────


@router.get("/access", summary="个人系统设置访问探针（不抛 403）")
async def system_settings_access(
    user: Optional[UserContext] = Depends(require_auth(False)),
    db: Session = Depends(get_db),
):
    """前端据此显隐「设置 → 系统管理」入口；只返回布尔与版本，不泄露配置内容。"""
    allowed = bool(user) and user_can_manage_system_settings(db, user.user_id)
    return success_response(data={"allowed": allowed, "edition": settings.edition.edition})


# ── Service configs (whitelist groups) ───────────────────────────────────────────────────


@router.get("/service-configs", summary="列出个人相关服务配置（白名单分组）")
async def list_personal_configs(_: str = Depends(require_system_settings)):
    """按分组返回白名单内的服务配置项；密钥类配置已脱敏。"""
    configs = _svc().get_all_configs()
    personal_groups = _personal_groups()
    grouped: Dict[str, dict] = {}
    for gk, label in personal_groups.items():
        grouped[gk] = {
            "group_key": gk,
            "label": label,
            "testable": gk in _TESTABLE_GROUPS,
            "items": [],
        }
    for cfg in configs:
        gk = cfg.get("group_key")
        if gk in grouped:
            grouped[gk]["items"].append(config_row_to_dict(cfg))
    return success_response(data=list(grouped.values()))


@router.put("/service-configs", summary="批量更新个人相关服务配置")
async def update_personal_configs(
    body: BulkUpdateRequest,
    _: str = Depends(require_system_settings),
):
    """按 key 批量写入；key 必须落在白名单分组内，越界返回 400。

    写入后异步重建 MCP 连接池，让搜索等 MCP 子进程拿到新 key（≤30s 生效）。
    掩码值（含 ``****``）由底层 ``bulk_set`` 跳过，防止把脱敏串写回真实密钥。
    """
    if not body.items:
        raise HTTPException(status_code=400, detail="items cannot be empty")
    svc = _svc()
    personal_groups = _personal_groups()
    key_to_group = {c["config_key"]: c.get("group_key") for c in svc.get_all_configs()}
    for item in body.items:
        group = key_to_group.get(item.key.strip())
        if group not in personal_groups:
            raise HTTPException(
                status_code=400,
                detail=f"配置项 {item.key} 不在个人可配置范围内",
            )
    svc.bulk_set([item.model_dump() for item in body.items], updated_by="me_system")
    asyncio.ensure_future(reinitialize_mcp_pool("me_system"))
    return success_response(data={"updated": len(body.items)})


@router.post("/service-configs/test/{group_key}", summary="测试服务连通性")
async def test_personal_config(
    group_key: str,
    _: str = Depends(require_system_settings),
):
    """Test connectivity for an allowed service-config group.

    Internet search tests only the API key that belongs to the selected engine.
    """
    if group_key not in _personal_groups():
        raise HTTPException(status_code=404, detail=f"Unknown group: {group_key}")
    if group_key not in _TESTABLE_GROUPS:
        raise HTTPException(status_code=400, detail="该分组无可测试的外部服务")
    return success_response(data=await test_service_group(group_key))


# ── Tenant quota (M14, phase 1) ───────────────────────────────────────────

_TENANT_QUOTA_KEY = "tenant_quota"


@router.get("/tenant-quota", summary="读取租户配额（管理面）")
async def get_tenant_quota(_: str = Depends(require_system_settings)):
    """``{"<tenant_id>": {"max_users": int, "daily_chat_runs_per_user": int}}``；缺席键 = 无限制。"""
    from core.db.engine import SessionLocal
    from core.db.models import SystemConfig

    with SessionLocal() as db:
        row = db.get(SystemConfig, _TENANT_QUOTA_KEY)
    if row is None or not (row.config_value or "").strip():
        return success_response(data={})
    try:
        payload = json.loads(row.config_value)
    except json.JSONDecodeError:
        payload = {}
    return success_response(data=payload if isinstance(payload, dict) else {})


@router.put("/tenant-quota", summary="写入租户配额（管理面；热生效，缺席键=无限制）")
async def put_tenant_quota(body: Dict[str, dict], _: str = Depends(require_system_settings)):
    """整体替换配额表。每租户仅认 ``max_users`` / ``daily_chat_runs_per_user`` 两个正整数旋钮。"""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object keyed by tenant_id")
    cleaned: Dict[str, dict] = {}
    for tenant_id, knobs in body.items():
        tid = str(tenant_id or "").strip()
        if not tid:
            raise HTTPException(status_code=400, detail="tenant_id cannot be empty")
        if not isinstance(knobs, dict):
            raise HTTPException(status_code=400, detail=f"{tid}: knobs must be an object")
        entry: Dict[str, int] = {}
        for knob in ("max_users", "daily_chat_runs_per_user"):
            if knob not in knobs or knobs[knob] is None:
                continue
            try:
                value = int(knobs[knob])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"{tid}.{knob} 必须是整数")
            if value < 0:
                raise HTTPException(status_code=400, detail=f"{tid}.{knob} 不能为负")
            entry[knob] = value
        cleaned[tid] = entry

    from core.db.engine import SessionLocal
    from core.db.models import SystemConfig

    with SessionLocal() as db:
        row = db.get(SystemConfig, _TENANT_QUOTA_KEY)
        if row is None:
            row = SystemConfig(
                config_key=_TENANT_QUOTA_KEY,
                display_name="租户配额",
                description="租户级账号与对话运行配额（M14）",
                group_key="tenant_quota",
                is_secret=False,
            )
            db.add(row)
        row.config_value = json.dumps(cleaned, ensure_ascii=False)
        db.commit()
    return success_response(data=cleaned)


# ── Middleware overrides (P1.1, registry) ─────────────────────────────────


@router.get("/middleware-config", summary="读取中间件装配视图（管理面）")
async def get_middleware_config(_: str = Depends(require_system_settings)):
    """每项 ``{name, level, enabled}``。security 级恒启用（注册表拒绝禁用）。"""
    from core.llm.middleware_registry import (
        DEFAULT_REGISTRY,
        LEVEL_SECURITY,
        load_disabled_names,
    )

    disabled = load_disabled_names()
    return success_response(
        data={
            "items": [
                {
                    "name": spec.name,
                    "level": spec.level,
                    "enabled": spec.name not in disabled,
                    "locked": spec.level == LEVEL_SECURITY,
                }
                for spec in DEFAULT_REGISTRY
            ]
        }
    )


@router.put("/middleware-config", summary="切换可禁用的功能中间件（管理面；security 级拒绝）")
async def put_middleware_config(body: Dict[str, list], _: str = Depends(require_system_settings)):
    """整体替换 ``disabled`` 名单。仅接受 functional 名单内的名字；security 级与未知名一律 400。"""
    if not isinstance(body, dict) or not isinstance(body.get("disabled"), list):
        raise HTTPException(status_code=400, detail='body must be {"disabled": [names]}')

    from core.llm.middleware_registry import (
        LEVEL_FUNCTIONAL,
        LEVEL_SECURITY,
        OVERRIDES_CONFIG_KEY,
        REGISTRY_BY_NAME,
    )

    disabled: List[str] = []
    for name in body["disabled"]:
        spec = REGISTRY_BY_NAME.get(str(name))
        if spec is None:
            raise HTTPException(status_code=400, detail=f"未知中间件: {name}")
        if spec.level == LEVEL_SECURITY:
            raise HTTPException(status_code=400, detail=f"安全中间件不可禁用: {name}")
        if spec.level == LEVEL_FUNCTIONAL:
            disabled.append(spec.name)

    from core.db.engine import SessionLocal
    from core.db.models import SystemConfig

    with SessionLocal() as db:
        row = db.get(SystemConfig, OVERRIDES_CONFIG_KEY)
        if row is None:
            row = SystemConfig(
                config_key=OVERRIDES_CONFIG_KEY,
                display_name="中间件装配覆盖",
                description="功能中间件禁用名单（P1.1 注册表）",
                group_key="middleware",
                is_secret=False,
            )
            db.add(row)
        row.config_value = json.dumps({"disabled": disabled}, ensure_ascii=False)
        db.commit()
    return success_response(data={"disabled": disabled})


# ── Implementation registry (P1.2: judge / audit sink) ────────────────────


@router.get("/impl-registry", summary="读取 Judge/审计账本实现选择（管理面）")
async def get_impl_registry(_: str = Depends(require_system_settings)):
    from core.safeharness_registry import (
        AUDIT_SINK_IMPLEMENTATIONS,
        JUDGE_IMPLEMENTATIONS,
        load_selection,
    )

    selection = load_selection()
    return success_response(
        data={
            "selection": selection,
            "judge_provider": list(JUDGE_IMPLEMENTATIONS),
            "audit_sink": list(AUDIT_SINK_IMPLEMENTATIONS),
        }
    )


@router.put("/impl-registry", summary="写入 Judge/审计账本实现选择（管理面；未知名 400）")
async def put_impl_registry(body: Dict[str, str], _: str = Depends(require_system_settings)):
    from core.safeharness_registry import (
        _known,
        IMPL_REGISTRY_CONFIG_KEY,
    )

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    judge = str(body.get("judge_provider") or "").strip()
    sink = str(body.get("audit_sink") or "").strip()
    if judge and judge not in _known("judge"):
        raise HTTPException(status_code=400, detail=f"未知 judge_provider: {judge}")
    if sink and sink not in _known("audit"):
        raise HTTPException(status_code=400, detail=f"未知 audit_sink: {sink}")

    payload = {"judge_provider": judge or "openai_compatible", "audit_sink": sink or "auto"}
    from core.db.engine import SessionLocal
    from core.db.models import SystemConfig

    with SessionLocal() as db:
        row = db.get(SystemConfig, IMPL_REGISTRY_CONFIG_KEY)
        if row is None:
            row = SystemConfig(
                config_key=IMPL_REGISTRY_CONFIG_KEY,
                display_name="SafeHarness 实现注册表选择",
                description="Judge 与审计账本实现选择（P1.2）",
                group_key="safeharness",
                is_secret=False,
            )
            db.add(row)
        row.config_value = json.dumps(payload, ensure_ascii=False)
        db.commit()
    return success_response(data=payload)
