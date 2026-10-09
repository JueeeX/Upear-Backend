"""Deterministic validation for AgentCardV1 and FSMPlanV1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from orchestration.subagents.fsm_contracts import (
    AgentCardV1,
    CapabilityV1,
    ContractError,
    FSMPlanV1,
    FSMStateV1,
    FSMTransitionV1,
    agent_registry_hash,
    canonical_json,
    fsm_plan_hash,
    parse_agent_card,
    parse_fsm,
)


_GENERIC_AGENT_OUTCOMES = frozenset({"failure", "timeout"})
_APPROVAL_OUTCOMES = frozenset({"approved", "rejected", "expired", "failure", "timeout"})
_MAX_TASK_INPUTS = 128


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class FSMValidationResult:
    valid: bool
    issues: tuple[ValidationIssue, ...]
    fsm: FSMPlanV1 | None = None
    plan_hash: str | None = None
    agent_registry_hash: str | None = None

    @property
    def status(self) -> str:
        return "verified" if self.valid else "invalid"

    def evidence(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
            "checks": [
                "contract",
                "registry",
                "bindings",
                "task_input_values",
                "graph",
                "failure_timeout_paths",
                "approval_paths",
            ],
        }


class _IssueCollector:
    def __init__(self) -> None:
        self.items: list[ValidationIssue] = []

    def add(self, code: str, path: str, message: str) -> None:
        self.items.append(ValidationIssue(code, path, message))


def _schema_properties(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    properties = schema.get("properties")
    return properties if isinstance(properties, Mapping) else {}


def _schema_required(schema: Mapping[str, Any]) -> set[str]:
    required = schema.get("required")
    return set(required) if isinstance(required, list) else set()


def _schema_type(schema: Any) -> str | None:
    if not isinstance(schema, Mapping):
        return None
    value = schema.get("type")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    return None


def _types_compatible(source: Any, target: Any) -> bool:
    source_type = _schema_type(source)
    target_type = _schema_type(target)
    if source_type is None or target_type is None:
        return False
    if source_type == target_type:
        return True
    return source_type == "integer" and target_type == "number"


def _json_type_matches(value: Any, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, Mapping)
    if type_name == "null":
        return value is None
    return True


def _task_input_conforms(value: Any, schema: Mapping[str, Any]) -> bool:
    """Instance-level check on a model-provided task input value.

    Enforces only constraints the capability's input schema actually declares
    (enum, type); anything undeclared stays the handler's runtime concern.
    This is what stops e.g. severity="high" from surviving plan verification
    when the capability only accepts critical/major/minor/warning.
    """
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        return False
    type_name = _schema_type(schema)
    return type_name is None or _json_type_matches(value, type_name)


def _edges(transitions: Iterable[FSMTransitionV1]) -> dict[str, list[FSMTransitionV1]]:
    result: dict[str, list[FSMTransitionV1]] = {}
    for transition in transitions:
        result.setdefault(transition.from_state, []).append(transition)
    return result


def _reachable(start: str, outgoing: Mapping[str, list[FSMTransitionV1]], *, blocked: str | None = None) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current == blocked or current in seen:
            continue
        seen.add(current)
        stack.extend(edge.to_state for edge in outgoing.get(current, ()))
    return seen


def _has_cycle(state_ids: set[str], outgoing: Mapping[str, list[FSMTransitionV1]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(state_id: str) -> bool:
        if state_id in visiting:
            return True
        if state_id in visited:
            return False
        visiting.add(state_id)
        if any(visit(edge.to_state) for edge in outgoing.get(state_id, ())):
            return True
        visiting.remove(state_id)
        visited.add(state_id)
        return False

    return any(visit(state_id) for state_id in state_ids)


def _all_can_reach_terminal(
    state_ids: set[str], terminal_ids: set[str], transitions: Iterable[FSMTransitionV1]
) -> set[str]:
    reverse: dict[str, set[str]] = {}
    for transition in transitions:
        reverse.setdefault(transition.to_state, set()).add(transition.from_state)
    result = set(terminal_ids)
    stack = list(terminal_ids)
    while stack:
        current = stack.pop()
        for predecessor in reverse.get(current, ()):
            if predecessor not in result:
                result.add(predecessor)
                stack.append(predecessor)
    return state_ids - result


def _registry_map(
    cards: Iterable[AgentCardV1 | Mapping[str, Any]], issues: _IssueCollector
) -> dict[str, AgentCardV1]:
    result: dict[str, AgentCardV1] = {}
    for index, raw in enumerate(cards):
        try:
            card = parse_agent_card(raw, f"agent_registry[{index}]")
        except ContractError as exc:
            issues.add("invalid_agent_card", f"agent_registry[{index}]", str(exc))
            continue
        if card.agent_id in result:
            issues.add("duplicate_agent", f"agent_registry[{index}].agent_id", card.agent_id)
        else:
            result[card.agent_id] = card
    return result


def _state_capability(
    state: FSMStateV1,
    registry: Mapping[str, AgentCardV1],
    issues: _IssueCollector,
) -> tuple[AgentCardV1 | None, CapabilityV1 | None]:
    if state.agent_id not in registry:
        issues.add("unknown_agent", f"states.{state.state_id}.agent_id", str(state.agent_id))
        return None, None
    card = registry[state.agent_id]
    capability = card.capability(state.capability_id or "")
    if capability is None:
        issues.add(
            "unknown_capability",
            f"states.{state.state_id}.capability_id",
            f"{state.agent_id}:{state.capability_id}",
        )
    return card, capability


def _validate_bindings(
    fsm: FSMPlanV1,
    states: Mapping[str, FSMStateV1],
    capabilities: Mapping[str, CapabilityV1],
    outgoing: Mapping[str, list[FSMTransitionV1]],
    issues: _IssueCollector,
) -> None:
    task = fsm.task
    task_inputs = task.get("inputs") if isinstance(task, Mapping) else None
    if not isinstance(task_inputs, Mapping):
        issues.add("missing_task_inputs", "task.inputs", "task.inputs must be an object")
        task_inputs = {}
    if len(task_inputs) > _MAX_TASK_INPUTS:
        issues.add("size_limit", "task.inputs", "too many task inputs")

    reachable_from_initial = _reachable(fsm.initial_state, outgoing)
    for state_id, state in states.items():
        if state.kind != "agent":
            continue
        capability = capabilities.get(state_id)
        if capability is None:
            continue
        properties = _schema_properties(capability.input_schema)
        required = _schema_required(capability.input_schema)
        bindings = state.input_bindings
        for key in bindings:
            if key not in properties:
                issues.add("unknown_input", f"states.{state_id}.input_bindings.{key}", key)
        for key in required:
            if key not in bindings:
                issues.add("missing_input_binding", f"states.{state_id}.input_bindings.{key}", key)
        for key, binding in bindings.items():
            target_schema = properties.get(key)
            if target_schema is None:
                continue
            if binding.source == "task":
                if binding.key not in task_inputs:
                    issues.add("missing_task_input", f"states.{state_id}.input_bindings.{key}", binding.key)
                    continue
                if not _task_input_conforms(task_inputs[binding.key], target_schema):
                    issues.add(
                        "invalid_task_input_value",
                        f"states.{state_id}.input_bindings.{key}",
                        binding.key,
                    )
                continue
            source_state = states.get(binding.state_id or "")
            if source_state is None:
                issues.add("unknown_binding_state", f"states.{state_id}.input_bindings.{key}", binding.state_id or "")
                continue
            if source_state.kind != "agent":
                issues.add("invalid_binding_source", f"states.{state_id}.input_bindings.{key}", source_state.state_id)
                continue
            if source_state.state_id not in reachable_from_initial:
                issues.add("unreachable_binding_source", f"states.{state_id}.input_bindings.{key}", source_state.state_id)
            if state_id not in _reachable(source_state.state_id, outgoing):
                issues.add("non_predecessor_binding", f"states.{state_id}.input_bindings.{key}", source_state.state_id)
            source_capability = capabilities.get(source_state.state_id)
            if source_capability is None:
                continue
            source_schema = _schema_properties(source_capability.output_schema).get(binding.key)
            if source_schema is None:
                issues.add("unknown_output_binding", f"states.{state_id}.input_bindings.{key}", binding.key)
            elif not _types_compatible(source_schema, target_schema):
                issues.add("schema_type_mismatch", f"states.{state_id}.input_bindings.{key}", binding.key)


def _validate_graph(
    fsm: FSMPlanV1,
    states: Mapping[str, FSMStateV1],
    outgoing: Mapping[str, list[FSMTransitionV1]],
    issues: _IssueCollector,
) -> None:
    state_ids = set(states)
    if fsm.initial_state not in states:
        issues.add("unknown_initial_state", "initial_state", fsm.initial_state)
        return
    terminal_ids = set(fsm.terminal_states)
    if not terminal_ids:
        issues.add("missing_terminal_state", "terminal_states", "at least one terminal is required")
    if len(terminal_ids) != len(fsm.terminal_states):
        issues.add("duplicate_terminal_state", "terminal_states", "duplicate terminal")
    for state_id in terminal_ids:
        if state_id not in states:
            issues.add("unknown_terminal_state", "terminal_states", state_id)
        elif states[state_id].kind != "terminal":
            issues.add("terminal_kind_mismatch", f"states.{state_id}", "terminal_states must be kind=terminal")
    for state_id, state in states.items():
        if state.kind == "terminal" and state_id not in terminal_ids:
            issues.add("terminal_not_declared", f"states.{state_id}", state_id)
        if state.kind == "terminal" and outgoing.get(state_id):
            issues.add("terminal_has_transition", f"states.{state_id}", state_id)
        if state.kind != "terminal" and not outgoing.get(state_id):
            issues.add("dead_end_state", f"states.{state_id}", state_id)
    reachable = _reachable(fsm.initial_state, outgoing)
    for state_id in state_ids - reachable:
        issues.add("unreachable_state", f"states.{state_id}", state_id)
    if _has_cycle(state_ids, outgoing):
        issues.add("cycle", "transitions", "FSM must be acyclic")
    for state_id in _all_can_reach_terminal(state_ids, terminal_ids, fsm.transitions):
        issues.add("no_terminal_path", f"states.{state_id}", state_id)


def _validate_transitions(
    fsm: FSMPlanV1,
    states: Mapping[str, FSMStateV1],
    capabilities: Mapping[str, CapabilityV1],
    issues: _IssueCollector,
) -> dict[str, list[FSMTransitionV1]]:
    outgoing: dict[str, list[FSMTransitionV1]] = {}
    seen: set[tuple[str, str]] = set()
    for index, transition in enumerate(fsm.transitions):
        path = f"transitions[{index}]"
        if transition.from_state not in states:
            issues.add("unknown_transition_source", f"{path}.from_state", transition.from_state)
            continue
        if transition.to_state not in states:
            issues.add("unknown_transition_target", f"{path}.to_state", transition.to_state)
            continue
        key = (transition.from_state, transition.condition)
        if key in seen:
            issues.add("duplicate_transition", path, f"{key[0]}:{key[1]}")
        seen.add(key)
        state = states[transition.from_state]
        if state.kind == "agent":
            allowed = set(capabilities.get(state.state_id).outcomes if capabilities.get(state.state_id) else ())
            allowed.update(_GENERIC_AGENT_OUTCOMES)
        elif state.kind == "approval":
            allowed = set(_APPROVAL_OUTCOMES)
        else:
            allowed = set()
        if transition.condition not in allowed:
            issues.add("invalid_transition_condition", f"{path}.condition", transition.condition)
        outgoing.setdefault(transition.from_state, []).append(transition)
    for state_id, state in states.items():
        if state.kind == "agent":
            conditions = {item.condition for item in outgoing.get(state_id, ())}
            for required in _GENERIC_AGENT_OUTCOMES:
                if required not in conditions:
                    issues.add("missing_failure_timeout_path", f"states.{state_id}", required)
        elif state.kind == "approval":
            conditions = {item.condition for item in outgoing.get(state_id, ())}
            for required in ("approved", "rejected", "expired"):
                if required not in conditions:
                    issues.add("missing_approval_path", f"states.{state_id}", required)
    return outgoing


def _validate_approvals(
    fsm: FSMPlanV1,
    states: Mapping[str, FSMStateV1],
    registry: Mapping[str, AgentCardV1],
    outgoing: Mapping[str, list[FSMTransitionV1]],
    issues: _IssueCollector,
) -> None:
    approvals_by_target: dict[str, list[str]] = {}
    for state in states.values():
        if state.kind == "approval":
            if state.approval_for not in states:
                issues.add("unknown_approval_target", f"states.{state.state_id}.approval_for", state.approval_for or "")
                continue
            target = states[state.approval_for]
            if target.kind != "agent":
                issues.add("invalid_approval_target", f"states.{state.state_id}.approval_for", target.state_id)
            else:
                approvals_by_target.setdefault(target.state_id, []).append(state.state_id)

    initial = fsm.initial_state
    for target_id, state in states.items():
        if state.kind != "agent" or state.agent_id not in registry or not state.capability_id:
            continue
        capability = registry[state.agent_id].capability(state.capability_id)
        if capability is None:
            continue
        requires_approval = capability.risk_level in {"high", "critical"} or capability.reversibility == "irreversible"
        if not requires_approval:
            continue
        approval_ids = approvals_by_target.get(target_id, [])
        if not approval_ids:
            issues.add("high_risk_without_approval", f"states.{target_id}", target_id)
            continue
        for approval_id in approval_ids:
            if target_id in _reachable(initial, outgoing, blocked=approval_id):
                issues.add("approval_bypass", f"states.{target_id}", approval_id)


def validate_fsm(
    candidate: FSMPlanV1 | Mapping[str, Any],
    agent_registry: Iterable[AgentCardV1 | Mapping[str, Any]],
) -> FSMValidationResult:
    """Return a verified FSM only when every deterministic check succeeds."""

    issues = _IssueCollector()
    registry = _registry_map(agent_registry, issues)
    try:
        fsm = parse_fsm(candidate)
    except ContractError as exc:
        issues.add("invalid_fsm_contract", "fsm", str(exc))
        return FSMValidationResult(False, tuple(issues.items), None, None, agent_registry_hash(registry.values()))

    try:
        if len(canonical_json(fsm.to_dict()).encode("utf-8")) > 128 * 1024:
            issues.add("size_limit", "fsm", "serialized FSM exceeds 128 KiB")
    except (TypeError, ValueError):
        issues.add("invalid_json", "fsm", "FSM is not JSON serializable")

    state_ids = [state.state_id for state in fsm.states]
    states = {state.state_id: state for state in fsm.states}
    if len(state_ids) != len(states):
        issues.add("duplicate_state", "states", "state_id must be unique")
    capabilities: dict[str, CapabilityV1] = {}
    for state in fsm.states:
        if state.kind == "agent":
            _, capability = _state_capability(state, registry, issues)
            if capability is not None:
                capabilities[state.state_id] = capability
    outgoing = _validate_transitions(fsm, states, capabilities, issues)
    _validate_graph(fsm, states, outgoing, issues)
    _validate_bindings(fsm, states, capabilities, outgoing, issues)
    _validate_approvals(fsm, states, registry, outgoing, issues)

    registry_hash = agent_registry_hash(registry.values())
    if issues.items:
        return FSMValidationResult(False, tuple(issues.items), None, None, registry_hash)

    plan_hash = fsm_plan_hash(fsm)
    verified = FSMPlanV1(
        task=fsm.task,
        initial_state=fsm.initial_state,
        states=fsm.states,
        transitions=fsm.transitions,
        terminal_states=fsm.terminal_states,
        validation={"status": "verified"},
        plan_hash=plan_hash,
        agent_registry_hash=registry_hash,
    )
    return FSMValidationResult(True, (), verified, plan_hash, registry_hash)
