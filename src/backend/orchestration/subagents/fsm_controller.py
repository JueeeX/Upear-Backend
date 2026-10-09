"""Deterministic FSM controller with resumable, single-state execution."""

from __future__ import annotations

import inspect
import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping
from uuid import uuid4

from orchestration.subagents.fsm_contracts import FSMPlanV1, FSMStateV1, fsm_plan_hash


MAX_OUTPUT_BYTES = 16 * 1024
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "system_prompt",
        "token",
    }
)


class ControllerError(RuntimeError):
    """Raised when checkpoint or transition invariants fail closed."""


def _safe_output(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[truncated]"
    if isinstance(value, Mapping):
        return {
            str(key): _safe_output(item, depth=depth + 1)
            for key, item in value.items()
            if str(key).lower() not in _SECRET_KEYS
        }
    if isinstance(value, list):
        return [_safe_output(item, depth=depth + 1) for item in value[:128]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:1000]


def _bounded_json(value: Any) -> dict[str, Any]:
    safe = _safe_output(value)
    if not isinstance(safe, dict):
        raise ControllerError("state output must be an object")
    encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ControllerError("state output exceeds size limit")
    return safe


@dataclass
class FSMRuntime:
    status: str
    current_state_id: str
    completed_state_ids: list[str] = field(default_factory=list)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    state_outcomes: dict[str, str] = field(default_factory=dict)
    state_runs: dict[str, dict[str, str]] = field(default_factory=dict)
    approval: dict[str, Any] | None = None
    last_error: str | None = None
    checkpoint_version: int = 0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], initial_state: str) -> "FSMRuntime":
        if not isinstance(value, Mapping):
            raise ControllerError("fsm_runtime must be an object")
        completed = value.get("completed_state_ids", [])
        outputs = value.get("outputs", {})
        outcomes = value.get("state_outcomes", {})
        state_runs = value.get("state_runs", {})
        if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
            raise ControllerError("completed_state_ids is invalid")
        if not isinstance(outputs, Mapping) or not isinstance(outcomes, Mapping) or not isinstance(state_runs, Mapping):
            raise ControllerError("fsm_runtime maps are invalid")
        return cls(
            status=str(value.get("status") or "running"),
            current_state_id=str(value.get("current_state_id") or initial_state),
            completed_state_ids=list(completed),
            outputs={str(key): _bounded_json(item) for key, item in outputs.items()},
            state_outcomes={str(key): str(item) for key, item in outcomes.items()},
            state_runs={
                str(key): {str(name): str(identifier) for name, identifier in item.items()}
                for key, item in state_runs.items()
                if isinstance(item, Mapping)
            },
            approval=dict(value["approval"]) if isinstance(value.get("approval"), Mapping) else None,
            last_error=str(value["last_error"]) if value.get("last_error") else None,
            checkpoint_version=int(value.get("checkpoint_version") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "current_state_id": self.current_state_id,
            "completed_state_ids": list(self.completed_state_ids),
            "outputs": dict(self.outputs),
            "state_outcomes": dict(self.state_outcomes),
            "state_runs": dict(self.state_runs),
            "approval": dict(self.approval) if self.approval else None,
            "last_error": self.last_error,
            "checkpoint_version": self.checkpoint_version,
        }


@dataclass(frozen=True)
class StateExecutionContext:
    parent_run_id: str
    state_id: str
    agent_id: str
    capability_id: str
    session_id: str
    run_id: str
    delegation_id: str
    trusted_intent: str | None = None


Handler = Callable[[Mapping[str, Any], StateExecutionContext], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


class FSMController:
    """Run one verified FSM; all decisions come from static transition labels."""

    def __init__(
        self,
        fsm: FSMPlanV1,
        *,
        parent_run_id: str,
        agent_registry_hash: str,
        handlers: Mapping[tuple[str, str], Handler],
        runtime: FSMRuntime | Mapping[str, Any] | None = None,
        checkpoint: Callable[[FSMRuntime], None | Awaitable[None]] | None = None,
        state_timeouts: Mapping[tuple[str, str], float] | None = None,
        trusted_intent: str | None = None,
    ) -> None:
        if not parent_run_id:
            raise ControllerError("parent_run_id is required")
        if not fsm.plan_hash or fsm.plan_hash != fsm_plan_hash(fsm):
            raise ControllerError("FSM plan hash mismatch")
        if not fsm.agent_registry_hash or fsm.agent_registry_hash != agent_registry_hash:
            raise ControllerError("AgentCard registry hash mismatch")
        self.fsm = fsm
        self.parent_run_id = parent_run_id
        self.handlers = dict(handlers)
        self.checkpoint = checkpoint
        self.state_timeouts = dict(state_timeouts or {})
        self.trusted_intent = (
            trusted_intent.strip()
            if isinstance(trusted_intent, str) and trusted_intent.strip()
            else None
        )
        self.states = {state.state_id: state for state in fsm.states}
        self.transitions = {(item.from_state, item.condition): item for item in fsm.transitions}
        self.runtime = (
            runtime
            if isinstance(runtime, FSMRuntime)
            else FSMRuntime.from_dict(runtime, fsm.initial_state)
            if isinstance(runtime, Mapping)
            else FSMRuntime(status="running", current_state_id=fsm.initial_state)
        )
        self._verify_runtime()

    def _verify_runtime(self) -> None:
        if self.runtime.current_state_id not in self.states:
            raise ControllerError("checkpoint current state is not in FSM")
        if len(set(self.runtime.completed_state_ids)) != len(self.runtime.completed_state_ids):
            raise ControllerError("checkpoint contains duplicate completed states")
        if any(item not in self.states for item in self.runtime.completed_state_ids):
            raise ControllerError("checkpoint contains unknown completed state")

    async def _checkpoint(self) -> None:
        self.runtime.checkpoint_version += 1
        if self.checkpoint is None:
            return
        result = self.checkpoint(self.runtime)
        if inspect.isawaitable(result):
            await result

    def _inputs(self, state: FSMStateV1) -> dict[str, Any]:
        task_inputs = self.fsm.task.get("inputs")
        if not isinstance(task_inputs, Mapping):
            raise ControllerError("task.inputs is missing")
        result: dict[str, Any] = {}
        for input_key, binding in state.input_bindings.items():
            if binding.source == "task":
                if binding.key not in task_inputs:
                    raise ControllerError(f"missing task input: {binding.key}")
                result[input_key] = task_inputs[binding.key]
                continue
            source = self.runtime.outputs.get(binding.state_id or "")
            if source is None or binding.key not in source:
                raise ControllerError(f"missing state output: {binding.state_id}.{binding.key}")
            result[input_key] = source[binding.key]
        return result

    def _next(self, state_id: str, outcome: str) -> str:
        transition = self.transitions.get((state_id, outcome.lower()))
        if transition is None:
            raise ControllerError(f"no declared transition for {state_id}:{outcome}")
        return transition.to_state

    @staticmethod
    def _state_run_ids() -> dict[str, str]:
        return {
            "session_id": f"fsm_session_{uuid4().hex[:16]}",
            "run_id": f"fsm_run_{uuid4().hex[:16]}",
            "delegation_id": f"fsm_delegation_{uuid4().hex[:16]}",
        }

    async def run(self, approvals: Mapping[str, str] | None = None):
        approvals = approvals or {}
        while True:
            state = self.states[self.runtime.current_state_id]
            if state.kind == "terminal":
                self.runtime.status = "completed" if state.state_id == "complete" else "manual_review"
                await self._checkpoint()
                yield {"type": "plan_complete", "state_id": state.state_id, "status": self.runtime.status}
                return

            if state.kind == "approval":
                decision = approvals.get(state.state_id)
                approval = self.runtime.approval
                if approval is not None:
                    if (
                        approval.get("state_id") != state.state_id
                        or approval.get("plan_hash") != self.fsm.plan_hash
                    ):
                        raise ControllerError("checkpoint approval binding is invalid")
                    if approval.get("used") and decision is None:
                        raise ControllerError("checkpoint approval was already consumed")
                elif decision is not None:
                    raise ControllerError("approval decision has no issued approval")
                if decision is None:
                    if approval is not None and self._approval_expired(approval):
                        decision = "expired"
                    else:
                        self.runtime.status = "awaiting_approval"
                        approval_id = str(
                            (approval or {}).get("approval_id")
                            or f"approval_{uuid4().hex[:16]}"
                        )
                        self.runtime.approval = {
                            "approval_id": approval_id,
                            "state_id": state.state_id,
                            "approval_for": state.approval_for,
                            "run_id": self.parent_run_id,
                            "plan_hash": self.fsm.plan_hash,
                            "expires_at": (
                                datetime.now(timezone.utc) + timedelta(minutes=15)
                            ).isoformat(),
                            "used": False,
                        }
                        await self._checkpoint()
                        yield {"type": "plan_approval_required", **self.runtime.approval}
                        return
                if decision is not None:
                    decision = str(decision).lower()
                if decision not in {"approved", "rejected", "expired"}:
                    raise ControllerError("approval decision is invalid")
                if decision == "expired" and not self._approval_expired(approval or {}):
                    raise ControllerError("approval has not expired")
                resolved_approval = dict(approval or {})
                resolved_approval.update({"used": True, "decision": decision})
                self.runtime.approval = resolved_approval
                await self._checkpoint()
                yield {
                    "type": "plan_approval_resolved",
                    "approval_id": resolved_approval.get("approval_id"),
                    "state_id": state.state_id,
                    "decision": decision,
                }
                self.runtime.approval = None
                outcome = decision
            else:
                if state.state_id in self.runtime.completed_state_ids:
                    if not self.runtime.state_outcomes.get(state.state_id):
                        raise ControllerError("completed state has no recorded outcome")
                    outcome = self.runtime.state_outcomes[state.state_id]
                else:
                    yield {"type": "plan_state_start", "state_id": state.state_id, "agent_id": state.agent_id, "capability_id": state.capability_id}
                    ids = self.runtime.state_runs.get(state.state_id) or self._state_run_ids()
                    self.runtime.state_runs[state.state_id] = ids
                    handler = self.handlers.get((state.agent_id or "", state.capability_id or ""))
                    if handler is None:
                        output = {"outcome": "failure", "error": "capability handler unavailable"}
                    else:
                        context = StateExecutionContext(
                            parent_run_id=self.parent_run_id,
                            state_id=state.state_id,
                            agent_id=state.agent_id or "",
                            capability_id=state.capability_id or "",
                            trusted_intent=self.trusted_intent,
                            **ids,
                        )
                        try:
                            raw_output = handler(self._inputs(state), context)
                            timeout = self.state_timeouts.get(
                                (state.agent_id or "", state.capability_id or "")
                            )
                            if inspect.isawaitable(raw_output):
                                output = (
                                    await asyncio.wait_for(raw_output, timeout=timeout)
                                    if timeout
                                    else await raw_output
                                )
                            else:
                                output = raw_output
                            output = _bounded_json(output)
                        except asyncio.TimeoutError:
                            output = {
                                "outcome": "timeout",
                                "error": "state execution timed out",
                                "error_class": "TimeoutError",
                            }
                        except Exception as exc:
                            # Only the exception class travels onward; the message
                            # may embed tool payloads, so it stays in the server log.
                            output = {
                                "outcome": "failure",
                                "error": "state execution failed",
                                "error_class": type(exc).__name__,
                            }
                    outcome = str(output.get("outcome") or "failure").lower()
                    self.runtime.outputs[state.state_id] = output
                    self.runtime.state_outcomes[state.state_id] = outcome
                    if state.state_id not in self.runtime.completed_state_ids:
                        self.runtime.completed_state_ids.append(state.state_id)
                    await self._checkpoint()
                    yield {"type": "plan_state_complete", "state_id": state.state_id, "outcome": outcome, "output": output}

            next_state = self._next(state.state_id, outcome)
            self.runtime.current_state_id = next_state
            self.runtime.status = "running"
            await self._checkpoint()
            yield {"type": "plan_transition", "from_state": state.state_id, "condition": outcome, "to_state": next_state}

    @staticmethod
    def _approval_expired(approval: Mapping[str, Any]) -> bool:
        try:
            expires_at = datetime.fromisoformat(str(approval.get("expires_at")))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            return expires_at.astimezone(timezone.utc) <= datetime.now(timezone.utc)
        except (TypeError, ValueError):
            return True
