"""Stable, public contracts for constrained multi-agent plans.

This module deliberately contains no runtime handlers.  A contract identifies a
capability; the controller decides whether and how that capability executes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


AGENT_CARD_SCHEMA_VERSION = "agent-card.v1"
FSM_SCHEMA_VERSION = "fsm.v1"
MAX_FSM_BYTES = 128 * 1024
MAX_FSM_STATES = 64
MAX_FSM_TRANSITIONS = 256
MAX_SCHEMA_DEPTH = 12

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,95}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
_REVERSIBILITY = frozenset({"read_only", "reversible", "irreversible", "unknown"})
_STATE_KINDS = frozenset({"agent", "approval", "terminal"})
_BINDING_SOURCES = frozenset({"task", "state"})

_MODEL_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "credentials",
        "endpoint",
        "handler",
        "internal_handler",
        "password",
        "secret",
        "system_prompt",
        "systemprompt",
        "token",
    }
)


class ContractError(ValueError):
    """Raised when untrusted contract data cannot be parsed safely."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be an object")
    return value


def _string(value: Any, path: str, *, required: bool = True, max_length: int = 2048) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise ContractError(f"{path} must be a non-empty string")
    value = value.strip()
    if len(value) > max_length:
        raise ContractError(f"{path} exceeds {max_length} characters")
    return value


def _identifier(value: Any, path: str) -> str:
    value = _string(value, path, max_length=96)
    if not _ID_RE.fullmatch(value):
        raise ContractError(f"{path} has invalid identifier")
    return value


def _string_list(value: Any, path: str, *, required: bool = True, max_items: int = 64) -> tuple[str, ...]:
    if value is None and not required:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError(f"{path} must be an array")
    if len(value) > max_items:
        raise ContractError(f"{path} exceeds {max_items} items")
    result = tuple(_string(item, f"{path}[{index}]", max_length=256) for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ContractError(f"{path} contains duplicates")
    return result


def _json_value(value: Any, path: str, *, max_depth: int = MAX_SCHEMA_DEPTH, depth: int = 0) -> Any:
    if depth > max_depth:
        raise ContractError(f"{path} exceeds depth limit")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise ContractError(f"{path} exceeds object field limit")
        return {
            _string(key, f"{path}.<key>", max_length=256): _json_value(
                item, f"{path}.{key}", max_depth=max_depth, depth=depth + 1
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > 256:
            raise ContractError(f"{path} exceeds array item limit")
        return [
            _json_value(item, f"{path}[{index}]", max_depth=max_depth, depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise ContractError(f"{path} contains a non-JSON value")


def _schema(value: Any, path: str) -> dict[str, Any]:
    result = dict(_mapping(value, path))
    if result.get("type") not in (None, "object", "array", "string", "number", "integer", "boolean", "null"):
        raise ContractError(f"{path}.type is unsupported")
    properties = result.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        raise ContractError(f"{path}.properties must be an object")
    required = result.get("required")
    if required is not None:
        required_values = _string_list(required, f"{path}.required", max_items=128)
        if properties is not None and any(key not in properties for key in required_values):
            raise ContractError(f"{path}.required references an unknown property")
        result["required"] = list(required_values)
    return _json_value(result, path)


def _forbidden_key(value: Any, path: str = "agent_card") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _MODEL_FORBIDDEN_KEYS:
                return f"{path}.{key_text}"
            found = _forbidden_key(child, f"{path}.{key_text}")
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            found = _forbidden_key(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonicalize(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact(item)
            for key, item in value.items()
            if str(key).lower() not in _MODEL_FORBIDDEN_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_redact(item) for item in value]
    return value


@dataclass(frozen=True)
class CapabilityV1:
    capability_id: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    required_capabilities: tuple[str, ...]
    risk_level: str
    side_effects: tuple[str, ...]
    reversibility: str
    idempotent: bool
    timeout_seconds: int
    max_retries: int
    outcomes: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "capability") -> "CapabilityV1":
        raw = _mapping(value, path)
        capability_id = _identifier(raw.get("capability_id"), f"{path}.capability_id")
        risk_level = _string(raw.get("risk_level"), f"{path}.risk_level", max_length=16).lower()
        if risk_level not in _RISK_LEVELS:
            raise ContractError(f"{path}.risk_level is invalid")
        reversibility = _string(raw.get("reversibility"), f"{path}.reversibility", max_length=32).lower()
        if reversibility not in _REVERSIBILITY:
            raise ContractError(f"{path}.reversibility is invalid")
        idempotent = raw.get("idempotent")
        if not isinstance(idempotent, bool):
            raise ContractError(f"{path}.idempotent must be boolean")
        timeout_seconds = raw.get("timeout_seconds")
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 1 <= timeout_seconds <= 3600:
            raise ContractError(f"{path}.timeout_seconds is invalid")
        max_retries = raw.get("max_retries")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or not 0 <= max_retries <= 5:
            raise ContractError(f"{path}.max_retries is invalid")
        outcomes = _string_list(raw.get("outcomes"), f"{path}.outcomes", max_items=32)
        if not outcomes:
            raise ContractError(f"{path}.outcomes must not be empty")
        return cls(
            capability_id=capability_id,
            description=_string(raw.get("description", ""), f"{path}.description", required=False),
            input_schema=_schema(raw.get("input_schema"), f"{path}.input_schema"),
            output_schema=_schema(raw.get("output_schema"), f"{path}.output_schema"),
            required_capabilities=_string_list(
                raw.get("required_capabilities"), f"{path}.required_capabilities"
            ),
            risk_level=risk_level,
            side_effects=_string_list(raw.get("side_effects"), f"{path}.side_effects"),
            reversibility=reversibility,
            idempotent=idempotent,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            outcomes=outcomes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "required_capabilities": list(self.required_capabilities),
            "risk_level": self.risk_level,
            "side_effects": list(self.side_effects),
            "reversibility": self.reversibility,
            "idempotent": self.idempotent,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "outcomes": list(self.outcomes),
        }


@dataclass(frozen=True)
class AgentCardV1:
    agent_id: str
    runtime: str
    capabilities: tuple[CapabilityV1, ...]
    name: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "agent_card") -> "AgentCardV1":
        raw = _mapping(value, path)
        forbidden = _forbidden_key(raw, path)
        if forbidden:
            raise ContractError(f"{forbidden} is not allowed in a public AgentCard")
        if raw.get("schema_version", AGENT_CARD_SCHEMA_VERSION) != AGENT_CARD_SCHEMA_VERSION:
            raise ContractError(f"{path}.schema_version is invalid")
        capabilities = raw.get("capabilities")
        if not isinstance(capabilities, Sequence) or isinstance(capabilities, (str, bytes)):
            raise ContractError(f"{path}.capabilities must be an array")
        if not capabilities or len(capabilities) > 64:
            raise ContractError(f"{path}.capabilities has invalid size")
        parsed = tuple(
            CapabilityV1.from_dict(item, f"{path}.capabilities[{index}]")
            for index, item in enumerate(capabilities)
        )
        ids = [item.capability_id for item in parsed]
        if len(ids) != len(set(ids)):
            raise ContractError(f"{path}.capabilities contains duplicates")
        return cls(
            agent_id=_identifier(raw.get("agent_id"), f"{path}.agent_id"),
            runtime=_identifier(raw.get("runtime"), f"{path}.runtime"),
            capabilities=parsed,
            name=_string(raw.get("name", ""), f"{path}.name", required=False),
            description=_string(raw.get("description", ""), f"{path}.description", required=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AGENT_CARD_SCHEMA_VERSION,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "runtime": self.runtime,
            "capabilities": [item.to_dict() for item in self.capabilities],
        }

    def capability(self, capability_id: str) -> CapabilityV1 | None:
        return next((item for item in self.capabilities if item.capability_id == capability_id), None)


@dataclass(frozen=True)
class BindingV1:
    source: str
    key: str
    state_id: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "binding") -> "BindingV1":
        raw = _mapping(value, path)
        source = _string(raw.get("source"), f"{path}.source", max_length=16).lower()
        if source not in _BINDING_SOURCES:
            raise ContractError(f"{path}.source is invalid")
        key = _string(raw.get("key"), f"{path}.key", max_length=128)
        state_id = raw.get("state_id")
        if source == "state":
            state_id = _identifier(state_id, f"{path}.state_id")
        elif state_id is not None:
            raise ContractError(f"{path}.state_id is only valid for state bindings")
        return cls(source=source, key=key, state_id=state_id)

    def to_dict(self) -> dict[str, Any]:
        result = {"source": self.source, "key": self.key}
        if self.state_id is not None:
            result["state_id"] = self.state_id
        return result


@dataclass(frozen=True)
class FSMStateV1:
    state_id: str
    kind: str
    title: str = ""
    description: str = ""
    agent_id: str | None = None
    capability_id: str | None = None
    input_bindings: Mapping[str, BindingV1] = field(default_factory=dict)
    approval_for: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "state") -> "FSMStateV1":
        raw = _mapping(value, path)
        kind = _string(raw.get("kind"), f"{path}.kind", max_length=16).lower()
        if kind not in _STATE_KINDS:
            raise ContractError(f"{path}.kind is invalid")
        bindings = raw.get("input_bindings", {})
        if not isinstance(bindings, Mapping):
            raise ContractError(f"{path}.input_bindings must be an object")
        parsed_bindings = {
            _string(key, f"{path}.input_bindings.<key>", max_length=128): BindingV1.from_dict(
                binding, f"{path}.input_bindings.{key}"
            )
            for key, binding in bindings.items()
        }
        agent_id = raw.get("agent_id")
        capability_id = raw.get("capability_id")
        approval_for = raw.get("approval_for")
        if kind == "agent":
            agent_id = _identifier(agent_id, f"{path}.agent_id")
            capability_id = _identifier(capability_id, f"{path}.capability_id")
            if approval_for is not None:
                raise ContractError(f"{path}.approval_for is only valid for approval states")
        elif kind == "approval":
            approval_for = _identifier(approval_for, f"{path}.approval_for")
            if agent_id is not None or capability_id is not None or parsed_bindings:
                raise ContractError(f"{path} approval state contains agent fields")
        else:
            if agent_id is not None or capability_id is not None or approval_for is not None or parsed_bindings:
                raise ContractError(f"{path} terminal state contains non-terminal fields")
        return cls(
            state_id=_identifier(raw.get("state_id"), f"{path}.state_id"),
            kind=kind,
            title=_string(raw.get("title", ""), f"{path}.title", required=False, max_length=500),
            description=_string(
                raw.get("description", ""), f"{path}.description", required=False, max_length=2000
            ),
            agent_id=agent_id,
            capability_id=capability_id,
            input_bindings=parsed_bindings,
            approval_for=approval_for,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "state_id": self.state_id,
            "kind": self.kind,
        }
        if self.title:
            result["title"] = self.title
        if self.description:
            result["description"] = self.description
        if self.agent_id is not None:
            result["agent_id"] = self.agent_id
        if self.capability_id is not None:
            result["capability_id"] = self.capability_id
        if self.input_bindings:
            result["input_bindings"] = {
                key: self.input_bindings[key].to_dict() for key in sorted(self.input_bindings)
            }
        if self.approval_for is not None:
            result["approval_for"] = self.approval_for
        return result


@dataclass(frozen=True)
class FSMTransitionV1:
    from_state: str
    condition: str
    to_state: str
    description: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "transition") -> "FSMTransitionV1":
        raw = _mapping(value, path)
        condition = _string(raw.get("condition"), f"{path}.condition", max_length=64).lower()
        if not _ID_RE.fullmatch(condition):
            raise ContractError(f"{path}.condition has invalid identifier")
        return cls(
            from_state=_identifier(raw.get("from_state"), f"{path}.from_state"),
            condition=condition,
            to_state=_identifier(raw.get("to_state"), f"{path}.to_state"),
            description=_string(raw.get("description", ""), f"{path}.description", required=False),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "from_state": self.from_state,
            "condition": self.condition,
            "to_state": self.to_state,
        }
        if self.description:
            result["description"] = self.description
        return result


@dataclass(frozen=True)
class FSMPlanV1:
    task: Mapping[str, Any]
    initial_state: str
    states: tuple[FSMStateV1, ...]
    transitions: tuple[FSMTransitionV1, ...]
    terminal_states: tuple[str, ...]
    validation: Mapping[str, Any] | None = None
    plan_hash: str | None = None
    agent_registry_hash: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], path: str = "fsm") -> "FSMPlanV1":
        raw = _mapping(value, path)
        if raw.get("schema_version", FSM_SCHEMA_VERSION) != FSM_SCHEMA_VERSION:
            raise ContractError(f"{path}.schema_version is invalid")
        states = raw.get("states")
        transitions = raw.get("transitions")
        terminal_states = raw.get("terminal_states")
        if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
            raise ContractError(f"{path}.states must be an array")
        if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes)):
            raise ContractError(f"{path}.transitions must be an array")
        if not isinstance(terminal_states, Sequence) or isinstance(terminal_states, (str, bytes)):
            raise ContractError(f"{path}.terminal_states must be an array")
        if not states or len(states) > MAX_FSM_STATES:
            raise ContractError(f"{path}.states has invalid size")
        if len(transitions) > MAX_FSM_TRANSITIONS:
            raise ContractError(f"{path}.transitions has invalid size")
        parsed_states = tuple(
            FSMStateV1.from_dict(item, f"{path}.states[{index}]") for index, item in enumerate(states)
        )
        parsed_transitions = tuple(
            FSMTransitionV1.from_dict(item, f"{path}.transitions[{index}]")
            for index, item in enumerate(transitions)
        )
        parsed_terminal_states = _string_list(
            terminal_states, f"{path}.terminal_states", max_items=MAX_FSM_STATES
        )
        validation = raw.get("validation")
        if validation is not None and not isinstance(validation, Mapping):
            raise ContractError(f"{path}.validation must be an object")
        plan_hash = raw.get("plan_hash")
        registry_hash = raw.get("agent_registry_hash")
        for name, hash_value in (("plan_hash", plan_hash), ("agent_registry_hash", registry_hash)):
            if hash_value is not None and (not isinstance(hash_value, str) or not _HASH_RE.fullmatch(hash_value)):
                raise ContractError(f"{path}.{name} is invalid")
        return cls(
            task=_json_value(_mapping(raw.get("task"), f"{path}.task"), f"{path}.task"),
            initial_state=_identifier(raw.get("initial_state"), f"{path}.initial_state"),
            states=parsed_states,
            transitions=parsed_transitions,
            terminal_states=parsed_terminal_states,
            validation=_json_value(validation, f"{path}.validation") if validation is not None else None,
            plan_hash=plan_hash,
            agent_registry_hash=registry_hash,
        )

    def to_dict(self, *, include_hashes: bool = True, include_validation: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": FSM_SCHEMA_VERSION,
            "task": dict(self.task),
            "initial_state": self.initial_state,
            "states": [state.to_dict() for state in self.states],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "terminal_states": list(self.terminal_states),
        }
        if include_validation and self.validation is not None:
            result["validation"] = dict(self.validation)
        if include_hashes and self.plan_hash:
            result["plan_hash"] = self.plan_hash
        if include_hashes and self.agent_registry_hash:
            result["agent_registry_hash"] = self.agent_registry_hash
        return result


def parse_agent_card(value: AgentCardV1 | Mapping[str, Any], path: str = "agent_card") -> AgentCardV1:
    return value if isinstance(value, AgentCardV1) else AgentCardV1.from_dict(value, path)


def parse_fsm(value: FSMPlanV1 | Mapping[str, Any], path: str = "fsm") -> FSMPlanV1:
    if isinstance(value, FSMPlanV1):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("fsm"), Mapping):
        value = value["fsm"]
    return FSMPlanV1.from_dict(value, path)


def normalize_agent_card(value: AgentCardV1 | Mapping[str, Any]) -> dict[str, Any]:
    card = parse_agent_card(value)
    result = card.to_dict()
    result["capabilities"] = sorted(
        (
            {
                **capability,
                "required_capabilities": sorted(capability["required_capabilities"]),
                "side_effects": sorted(capability["side_effects"]),
                "outcomes": sorted(capability["outcomes"]),
            }
            for capability in result["capabilities"]
        ),
        key=lambda item: item["capability_id"],
    )
    return _canonicalize(result)


def _normalize_public_agent_card(value: AgentCardV1 | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a public projection without admitting runtime-only fields."""

    projected = _redact(value) if isinstance(value, Mapping) else value
    return normalize_agent_card(projected)


def agent_registry_hash(cards: Iterable[AgentCardV1 | Mapping[str, Any]]) -> str:
    normalized = sorted((_normalize_public_agent_card(card) for card in cards), key=lambda item: item["agent_id"])
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def agent_cards_model_view(cards: Iterable[AgentCardV1 | Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only public AgentCard fields; never expose runtime credentials."""

    return [_normalize_public_agent_card(card) for card in cards]


def normalize_fsm(value: FSMPlanV1 | Mapping[str, Any], *, include_hashes: bool = False) -> dict[str, Any]:
    fsm = parse_fsm(value)
    result = fsm.to_dict(include_hashes=include_hashes, include_validation=False)
    result["states"] = sorted(result["states"], key=lambda item: item["state_id"])
    result["transitions"] = sorted(
        result["transitions"],
        key=lambda item: (item["from_state"], item["condition"], item["to_state"]),
    )
    result["terminal_states"] = sorted(result["terminal_states"])
    return _canonicalize(result)


def fsm_plan_hash(value: FSMPlanV1 | Mapping[str, Any]) -> str:
    payload = normalize_fsm(value, include_hashes=False)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def default_agent_cards() -> tuple[AgentCardV1, ...]:
    """V1 public NOC registry; handlers remain server-side and are not here."""

    object_schema = lambda properties, required: {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    text = {"type": "string"}
    outcome = {"type": "string"}
    diagnosis = AgentCardV1(
        agent_id="noc-diagnosis-agent",
        name="NOC diagnosis agent",
        description="Read-only alarm query and diagnosis.",
        runtime="llm",
        capabilities=(
            CapabilityV1(
                capability_id="alarm.query",
                description="Query current alarms for a device.",
                input_schema=object_schema({"device_id": text}, ["device_id"]),
                output_schema=object_schema({"device_id": text, "alarms": {"type": "array"}, "outcome": outcome}, ["device_id", "alarms", "outcome"]),
                required_capabilities=("nms:alarms:read",),
                risk_level="low",
                side_effects=(),
                reversibility="read_only",
                idempotent=True,
                timeout_seconds=120,
                max_retries=1,
                outcomes=("success", "failure", "timeout"),
            ),
            CapabilityV1(
                capability_id="alarm.diagnose",
                description="Produce structured diagnosis from observed alarms.",
                input_schema=object_schema({"device_id": text, "alarms": {"type": "array"}}, ["device_id", "alarms"]),
                output_schema=object_schema({"outcome": outcome, "summary": text}, ["outcome", "summary"]),
                required_capabilities=("nms:alarms:read", "nms:devices:read"),
                risk_level="low",
                side_effects=(),
                reversibility="read_only",
                idempotent=True,
                timeout_seconds=120,
                max_retries=1,
                outcomes=("resolved", "needs_remediation", "manual_review", "failure", "timeout"),
            ),
            CapabilityV1(
                capability_id="alarm.verify",
                description="Verify whether the target alarm is resolved.",
                input_schema=object_schema({"device_id": text, "alarms": {"type": "array"}}, ["device_id", "alarms"]),
                output_schema=object_schema({"outcome": outcome, "summary": text}, ["outcome", "summary"]),
                required_capabilities=("nms:alarms:read", "nms:devices:read"),
                risk_level="low",
                side_effects=(),
                reversibility="read_only",
                idempotent=True,
                timeout_seconds=120,
                max_retries=1,
                outcomes=("resolved", "unresolved", "manual_review", "failure", "timeout"),
            ),
        ),
    )
    change = AgentCardV1(
        agent_id="config-change-agent",
        name="NOC configuration change agent",
        description="Simulated ticket creation and interface restart.",
        runtime="fixed_handler",
        capabilities=(
            CapabilityV1(
                capability_id="incident.create",
                description="Create a simulated incident ticket.",
                input_schema=object_schema(
                    {
                        "device_id": text,
                        "title": text,
                        "severity": {"type": "string", "enum": ["critical", "major", "minor", "warning"]},
                        "summary": text,
                    },
                    ["device_id", "title", "severity", "summary"],
                ),
                output_schema=object_schema({"ticket_id": text, "outcome": outcome}, ["ticket_id", "outcome"]),
                required_capabilities=("noc:tickets:write",),
                risk_level="medium",
                side_effects=("incident.create",),
                reversibility="reversible",
                idempotent=True,
                timeout_seconds=60,
                max_retries=0,
                outcomes=("success", "failure", "timeout"),
            ),
            CapabilityV1(
                capability_id="interface.restart",
                description="Restart a simulated network interface.",
                input_schema=object_schema(
                    {"device_id": text, "interface": text, "reason": text, "ticket_id": text},
                    ["device_id", "interface", "reason", "ticket_id"],
                ),
                output_schema=object_schema({"outcome": outcome, "summary": text}, ["outcome", "summary"]),
                required_capabilities=("nms:interface:restart",),
                risk_level="high",
                side_effects=("interface.restart",),
                reversibility="irreversible",
                idempotent=False,
                timeout_seconds=120,
                max_retries=0,
                outcomes=("success", "failure", "timeout", "manual_review"),
            ),
        ),
    )
    return diagnosis, change
