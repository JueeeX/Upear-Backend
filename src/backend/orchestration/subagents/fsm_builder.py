"""Deterministic FSM construction from a constrained capability proposal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from orchestration.subagents.fsm_contracts import AgentCardV1, CapabilityV1, parse_agent_card


@dataclass(frozen=True)
class FSMBuildIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class FSMBuildResult:
    fsm: dict[str, Any] | None
    issues: tuple[FSMBuildIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return self.fsm is not None and not self.issues


def _schema_properties(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    properties = schema.get("properties")
    return properties if isinstance(properties, Mapping) else {}


def _required_inputs(capability: CapabilityV1) -> tuple[str, ...]:
    required = capability.input_schema.get("required")
    return tuple(item for item in required if isinstance(item, str)) if isinstance(required, list) else ()


def _output_names(capability: CapabilityV1) -> tuple[str, ...]:
    return tuple(_schema_properties(capability.output_schema))


def _registry(
    cards: Iterable[AgentCardV1 | Mapping[str, Any]],
) -> dict[tuple[str, str], tuple[AgentCardV1, CapabilityV1]]:
    result: dict[tuple[str, str], tuple[AgentCardV1, CapabilityV1]] = {}
    for raw in cards:
        card = parse_agent_card(raw)
        for capability in card.capabilities:
            result[(card.agent_id, capability.capability_id)] = (card, capability)
    return result


def _proposal_items(proposal: Mapping[str, Any], issues: list[FSMBuildIssue]) -> list[tuple[str, str]]:
    raw_items = proposal.get("selected_capabilities")
    if not isinstance(raw_items, list) or not raw_items:
        issues.append(
            FSMBuildIssue(
                "missing_capability_selection",
                "selected_capabilities",
                "提案必须包含至少一个 selected_capabilities 项",
            )
        )
        return []
    result: list[tuple[str, str]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            issues.append(
                FSMBuildIssue("invalid_capability_selection", f"selected_capabilities[{index}]", "能力项必须是对象")
            )
            continue
        agent_id = item.get("agent_id")
        capability_id = item.get("capability_id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            issues.append(
                FSMBuildIssue(
                    "unknown_agent",
                    f"selected_capabilities[{index}].agent_id",
                    "缺少 agent_id",
                )
            )
            continue
        if not isinstance(capability_id, str) or not capability_id.strip():
            issues.append(
                FSMBuildIssue(
                    "unknown_capability",
                    f"selected_capabilities[{index}].capability_id",
                    "缺少 capability_id",
                )
            )
            continue
        result.append((agent_id.strip(), capability_id.strip()))
    return result


def _task(proposal: Mapping[str, Any], fallback_task: str | None, issues: list[FSMBuildIssue]) -> dict[str, Any]:
    raw_task = proposal.get("task")
    if raw_task is None:
        raw_task = {"goal": fallback_task or "多智能体任务", "inputs": {}}
    if not isinstance(raw_task, Mapping):
        issues.append(FSMBuildIssue("invalid_task", "task", "task 必须是对象"))
        return {"goal": fallback_task or "多智能体任务", "inputs": {}}
    task = dict(raw_task)
    if not isinstance(task.get("inputs", {}), Mapping):
        issues.append(FSMBuildIssue("invalid_task_inputs", "task.inputs", "task.inputs 必须是对象"))
        task["inputs"] = {}
    else:
        task["inputs"] = dict(task.get("inputs", {}))
    if not isinstance(task.get("goal", fallback_task or "多智能体任务"), str):
        issues.append(FSMBuildIssue("invalid_task_goal", "task.goal", "task.goal 必须是字符串"))
        task["goal"] = fallback_task or "多智能体任务"
    return task


def _find_producer(
    field: str,
    registry: Mapping[tuple[str, str], tuple[AgentCardV1, CapabilityV1]],
    selected: set[tuple[str, str]],
) -> tuple[str, str] | None:
    candidates = [
        key
        for key, (_, capability) in registry.items()
        if field in _output_names(capability) and key not in selected
    ]
    return min(candidates) if candidates else None


def _expand_dependencies(
    requested: list[tuple[str, str]],
    registry: Mapping[tuple[str, str], tuple[AgentCardV1, CapabilityV1]],
    task_inputs: set[str],
    issues: list[FSMBuildIssue],
) -> list[tuple[str, str]]:
    """Insert registered producers before consumers, with cycle-safe recursion."""

    ordered: list[tuple[str, str]] = []
    added: set[tuple[str, str]] = set()
    visiting: set[tuple[str, str]] = set()

    def add(item: tuple[str, str], path: str) -> None:
        if item in visiting:
            issues.append(
                FSMBuildIssue(
                    "capability_dependency_cycle",
                    path,
                    f"能力依赖存在循环: {item[0]}:{item[1]}",
                )
            )
            return
        if item in added:
            return
        pair = registry.get(item)
        if pair is None:
            issues.append(FSMBuildIssue("unknown_capability", path, f"未注册能力: {item[0]}:{item[1]}"))
            return
        visiting.add(item)
        _, capability = pair
        available = set(task_inputs)
        for existing in ordered:
            existing_capability = registry[existing][1]
            available.update(_output_names(existing_capability))
        for field in _required_inputs(capability):
            if field in available:
                continue
            producer = _find_producer(field, registry, added | visiting)
            if producer is None:
                issues.append(
                    FSMBuildIssue(
                        "missing_task_input",
                        f"{path}.input_bindings.{field}",
                        f"缺少输入 {field}，且注册表没有可用生产能力",
                    )
                )
                continue
            add(producer, f"{path}.dependency.{field}")
        visiting.remove(item)
        if item not in added:
            ordered.append(item)
            added.add(item)

    for index, item in enumerate(requested):
        add(item, f"selected_capabilities[{index}]")
    return ordered


def _default_target(outcome: str, next_state: str | None, complete: str, manual_review: str) -> str:
    normalized = outcome.lower()
    if normalized in {"failure", "timeout", "error", "manual_review", "unresolved", "rejected", "expired"}:
        return manual_review
    if normalized == "resolved":
        return complete
    return next_state or complete


def build_fsm_from_proposal(
    proposal: Mapping[str, Any],
    agent_registry: Iterable[AgentCardV1 | Mapping[str, Any]],
    *,
    fallback_task: str | None = None,
) -> FSMBuildResult:
    """Build all graph mechanics from a model-selected capability sequence."""

    issues: list[FSMBuildIssue] = []
    if not isinstance(proposal, Mapping):
        return FSMBuildResult(None, (FSMBuildIssue("invalid_proposal", "proposal", "提案必须是对象"),))
    try:
        registry = _registry(agent_registry)
    except Exception as exc:
        return FSMBuildResult(None, (FSMBuildIssue("invalid_agent_registry", "agent_registry", type(exc).__name__),))
    task = _task(proposal, fallback_task, issues)
    requested = _proposal_items(proposal, issues)
    selected = _expand_dependencies(requested, registry, set(task.get("inputs", {})), issues)
    if issues:
        return FSMBuildResult(None, tuple(issues))

    workers: list[dict[str, Any]] = []
    available: dict[str, str] = {key: "task" for key in task.get("inputs", {})}
    for index, item in enumerate(selected, start=1):
        card, capability = registry[item]
        state_id = f"s{index}"
        bindings: dict[str, dict[str, str]] = {}
        for field in _required_inputs(capability):
            source_state = available.get(field)
            if source_state == "task":
                bindings[field] = {"source": "task", "key": field}
            elif source_state:
                bindings[field] = {"source": "state", "state_id": source_state, "key": field}
            else:
                issues.append(
                    FSMBuildIssue(
                        "missing_task_input",
                        f"states.{state_id}.input_bindings.{field}",
                        f"缺少输入 {field}",
                    )
                )
        workers.append(
            {
                "state_id": state_id,
                "kind": "agent",
                "title": capability.description or capability.capability_id,
                "description": capability.description,
                "agent_id": card.agent_id,
                "capability_id": capability.capability_id,
                "input_bindings": bindings,
            }
        )
        for output_name in _output_names(capability):
            available[output_name] = state_id

    if issues:
        return FSMBuildResult(None, tuple(issues))

    complete = "complete"
    manual_review = "manual_review"
    route: list[dict[str, Any]] = []
    worker_entry: dict[str, str] = {}
    for worker in workers:
        _, capability = registry[(worker["agent_id"], worker["capability_id"])]
        requires_approval = (
            capability.risk_level in {"high", "critical"}
            or capability.reversibility == "irreversible"
        )
        if requires_approval:
            approval_id = f"approve_{worker['state_id']}"
            worker_entry[worker["state_id"]] = approval_id
            route.append(
                {
                    "state_id": approval_id,
                    "kind": "approval",
                    "title": f"审批：{worker['title']}",
                    "description": "高风险能力执行前需要人工审批。",
                    "approval_for": worker["state_id"],
                }
            )
        else:
            worker_entry[worker["state_id"]] = worker["state_id"]
        route.append(worker)

    states: list[dict[str, Any]] = []
    transitions: list[dict[str, str]] = []
    for index, worker in enumerate(workers):
        state_id = worker["state_id"]
        next_worker = workers[index + 1]["state_id"] if index + 1 < len(workers) else None
        next_target = worker_entry.get(next_worker, next_worker) if next_worker else complete
        _, capability = registry[(worker["agent_id"], worker["capability_id"])]
        for outcome in capability.outcomes:
            transitions.append(
                {
                    "from_state": state_id,
                    "condition": outcome,
                    "to_state": _default_target(outcome, next_target, complete, manual_review),
                }
            )
        states.append(worker)

    for item in route:
        if item["kind"] != "approval":
            continue
        worker_id = item["approval_for"]
        transitions.extend(
            [
                {"from_state": item["state_id"], "condition": "approved", "to_state": worker_id},
                {"from_state": item["state_id"], "condition": "rejected", "to_state": manual_review},
                {"from_state": item["state_id"], "condition": "expired", "to_state": manual_review},
            ]
        )
    states = route + [
        {"state_id": complete, "kind": "terminal", "title": "执行完成"},
        {"state_id": manual_review, "kind": "terminal", "title": "人工处理"},
    ]
    if states:
        first = next(item for item in states if item["kind"] != "terminal")
        initial_state = first["state_id"]
    else:
        initial_state = complete
    return FSMBuildResult(
        {
            "schema_version": "fsm.v1",
            "task": task,
            "initial_state": initial_state,
            "states": states,
            "transitions": transitions,
            "terminal_states": [complete, manual_review],
        },
        tuple(issues),
    )
