"""Judge role capability probe: fail-fast gate before role assignment.

The semantic judge is fail-closed — an endpoint that cannot answer the judge
contract blocks every protected request. These tests pin the probe's contract:
engine gate, JSON-contract success, non-JSON rejection, and transport failure.
"""

import json
from types import SimpleNamespace

import pytest

from api.routes.v1 import models as models_routes


def _provider(engine_provider="openai_compatible", provider_type="chat"):
    return SimpleNamespace(
        provider=engine_provider,
        provider_type=provider_type,
        base_url="http://127.0.0.1:9/v1",
        api_key="probe-key",
        model_name="probe-model",
    )


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _patch_client(monkeypatch, responses):
    """Queue is shared across client instantiations: the probe opens one client
    per attempt, and each attempt must consume the next queued response."""
    calls = {"count": 0, "payloads": []}

    class _RecordingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            calls["count"] += 1
            calls["payloads"].append(json)
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    monkeypatch.setattr(models_routes.httpx, "AsyncClient", lambda timeout: _RecordingClient())
    return calls


def _judge_payload(content):
    return {"choices": [{"message": {"content": content}}]}


_GOOD_CONTRACT = json.dumps({
    "label": "SAFE",
    "confidence": 0.9,
    "reason": "benign content",
})


@pytest.mark.asyncio
async def test_probe_rejects_non_openai_engine(monkeypatch):
    # The engine gate must short-circuit before any network request; if a request
    # is attempted anyway, the fake client raises.
    calls = _patch_client(monkeypatch, [AssertionError("no request expected")])

    result = await models_routes._probe_judge_role_capability(
        _provider(engine_provider="ollama")
    )

    assert result["ok"] is False
    assert "OpenAI 兼容" in result["error"]
    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_probe_accepts_valid_judge_contract(monkeypatch):
    calls = _patch_client(monkeypatch, [_FakeResponse(200, _judge_payload(_GOOD_CONTRACT))])

    result = await models_routes._probe_judge_role_capability(_provider())

    assert result["ok"] is True
    assert calls["count"] == 1
    # The first attempt mirrors the production caller: JSON response_format hint on.
    assert calls["payloads"][0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_probe_retries_without_json_hint_then_accepts(monkeypatch):
    responses = [
        _FakeResponse(400, text="response_format is not supported"),
        _FakeResponse(200, _judge_payload(_GOOD_CONTRACT)),
    ]
    calls = _patch_client(monkeypatch, responses)

    result = await models_routes._probe_judge_role_capability(_provider())

    assert result["ok"] is True
    assert calls["count"] == 2
    assert "response_format" not in calls["payloads"][1]


@pytest.mark.asyncio
async def test_probe_rejects_non_json_content(monkeypatch):
    # A 200 answer that violates the judge contract fails the probe without a
    # second attempt — dropping the response_format hint cannot turn prose into
    # the required JSON contract.
    calls = _patch_client(
        monkeypatch,
        [
            _FakeResponse(200, _judge_payload("I cannot produce JSON.")),
            _FakeResponse(200, _judge_payload("Still no JSON.")),
        ],
    )

    result = await models_routes._probe_judge_role_capability(_provider())

    assert result["ok"] is False
    assert calls["count"] == 1
    assert "判定契约" in result["error"]


@pytest.mark.asyncio
async def test_probe_rejects_transport_failure(monkeypatch):
    calls = _patch_client(monkeypatch, [ConnectionError("boom"), ConnectionError("boom")])

    result = await models_routes._probe_judge_role_capability(_provider())

    assert result["ok"] is False
    assert calls["count"] == 2
    assert "请求失败" in result["error"]
