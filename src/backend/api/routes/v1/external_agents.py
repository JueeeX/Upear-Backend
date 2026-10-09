"""External agent bridge API — run real GitHub agent frameworks, guarded.

The frameworks (smolagents / openai-agents / autogen) execute their own agent
loops with the platform's main model provider. Inbound messages pass the
SafeHarness structural checks first; blocks are audited like protected-chat
blocks. This surface is read/run only — it never registers anything into the
agent catalog or the evolution loop.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.backend import UserContext, get_current_user
from core.db.engine import get_db
from core.infra.exceptions import BadRequestError
from core.infra.responses import success_response
from api.deps import require_system_settings

from core.external_agents import available_external_agents, run_external_agent
from core.external_agents.model_settings import (
    get_model_settings,
    set_agent_model,
    set_default_model,
)
from core.external_agents.registry import CATALOG, ExternalAgentBlocked

router = APIRouter(prefix="/v1/external-agents", tags=["External Agents"])


@router.get("", summary="列出外部智能体（真实第三方框架接入）及其 AgentCard 与模型设置")
async def list_external_agents(user: UserContext = Depends(get_current_user), db: Session = Depends(get_db)):
    catalog = available_external_agents()
    settings = get_model_settings(db)

    from core.external_agents.cards import get_card
    from core.external_agents.registry import CATALOG as _CATALOG

    cards = {
        agent_id: get_card(db, agent_id, _CATALOG[agent_id])
        for agent_id in _CATALOG
    }
    return success_response(data={**catalog, "cards": cards, "model_settings": settings})


@router.get("/{agent_id}/card", summary="读取单个外部智能体的 A2A AgentCard")
async def get_external_agent_card(
    agent_id: str,
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if agent_id not in CATALOG:
        raise BadRequestError(f"未知外部智能体: {agent_id}")
    from core.external_agents.cards import get_card

    return success_response(data=get_card(db, agent_id, CATALOG[agent_id]))


class SupervisionConfigRequest(BaseModel):
    max_steps: int = Field(..., ge=1, le=50)
    loop_trigger: int = Field(..., ge=2, le=20)
    max_output_chars: int = Field(..., ge=200)
    max_total_chars: int = Field(..., ge=1000)


@router.get("/supervision-config", summary="读取外部智能体运行监督阈值")
async def get_supervision_config(
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from core.external_agents.supervisor import load_supervision_config

    return success_response(data=load_supervision_config(db).to_dict())


@router.put("/supervision-config", summary="设置外部智能体运行监督阈值（管理面）")
async def put_supervision_config(
    body: SupervisionConfigRequest,
    _: str = Depends(require_system_settings),
    db: Session = Depends(get_db),
):
    from core.external_agents.supervisor import (
        SupervisionConfig,
        save_supervision_config,
    )

    config = SupervisionConfig(
        max_steps=body.max_steps,
        loop_trigger=body.loop_trigger,
        max_output_chars=body.max_output_chars,
        max_total_chars=body.max_total_chars,
    )
    save_supervision_config(db, config)
    return success_response(data=config.to_dict())


class AgentCardRequest(BaseModel):
    card: dict = Field(..., description="A2A AgentCard JSON（camelCase 规范字段；未知字段保留）")


@router.put("/{agent_id}/card", summary="维护单个外部智能体的 A2A AgentCard（管理面）")
async def put_external_agent_card(
    agent_id: str,
    body: AgentCardRequest,
    _: str = Depends(require_system_settings),
    db: Session = Depends(get_db),
):
    if agent_id not in CATALOG:
        raise BadRequestError(f"未知外部智能体: {agent_id}")
    from core.external_agents.cards import save_card

    try:
        saved = save_card(db, agent_id, body.card)
    except ValueError as exc:
        raise BadRequestError(str(exc))
    return success_response(data=saved)


class ExternalAgentRunRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)


class ModelSettingRequest(BaseModel):
    scope: str = Field(..., description='"default" 或外部智能体 agent_id')
    provider_id: Optional[str] = Field(default=None, description='模型供应商 ID；null 表示清除该级设置')


class MemoryConfigRequest(BaseModel):
    history_window: Optional[int] = Field(default=None, ge=0, le=50, description="会话窗口条数")
    long_term_enabled: Optional[bool] = Field(default=None, description="跨会话长期记忆（mem0）开关")
    long_term_max_chars: Optional[int] = Field(default=None, ge=0, le=16000, description="长期记忆注入字符预算")
    long_term_limit: Optional[int] = Field(default=None, ge=1, le=20, description="长期记忆检索条数")


@router.put("/model-settings", summary="设置外部智能体模型（单智能体覆盖 / 统一默认）")
async def put_model_setting(
    body: ModelSettingRequest,
    _: str = Depends(require_system_settings),
    db: Session = Depends(get_db),
):
    if body.scope != "default" and body.scope not in CATALOG:
        raise BadRequestError(f"未知外部智能体: {body.scope}")
    try:
        if body.scope == "default":
            set_default_model(db, body.provider_id)
        else:
            set_agent_model(db, body.scope, body.provider_id)
    except ValueError as exc:
        raise BadRequestError(str(exc))
    return success_response(data=get_model_settings(db))


@router.get("/memory-config", summary="读取外部智能体记忆配置（会话窗口/长期记忆预算）")
async def get_memory_config(
    user: UserContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from core.external_agents.memory_settings import load_memory_config

    return success_response(data=load_memory_config(db).__dict__)


@router.put("/memory-config", summary="设置外部智能体记忆配置（管理面）")
async def put_memory_config(
    body: MemoryConfigRequest,
    _: str = Depends(require_system_settings),
    db: Session = Depends(get_db),
):
    from core.external_agents.memory_settings import save_memory_config

    patch = body.model_dump(exclude_none=True)
    merged = save_memory_config(db, patch)
    return success_response(data=merged.__dict__)


@router.post("/{agent_id}/run", summary="运行外部智能体（SafeHarness 结构检查前置）")
async def run_agent(
    agent_id: str,
    body: ExternalAgentRunRequest,
    user: UserContext = Depends(get_current_user),
):
    if agent_id not in CATALOG:
        raise BadRequestError(f"未知外部智能体: {agent_id}")
    context = {
        "tenant_id": getattr(user, "tenant_id", "") or "default",
        "session_id": f"ext-{user.user_id}",
        "trace_id": f"ext-{user.user_id}",
        "user_id": user.user_id,
    }
    try:
        data = await run_external_agent(agent_id, body.message, context=context)
    except ExternalAgentBlocked as exc:
        raise BadRequestError(
            f"SafeHarness 已拦截该请求（{exc.reason_code}）；外部智能体未执行。"
        )
    return success_response(data=data)


__all__ = ["router"]
