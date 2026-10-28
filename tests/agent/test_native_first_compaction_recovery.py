"""Offline turn-loop and durable-state controls for native-first admission.

Responses/checkpoints below are explicitly synthetic protocol fixtures, not live
provider acceptance. Historical usage/cost measurements live in the seam tests.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.agent.test_native_first_automatic_compaction import (
    HISTORICAL_PRESSURES, _agent, _tool_at_cost,
)
from tests.agent.test_run_agent_codex_responses import (
    _build_agent, _codex_message_response, _codex_tool_call_response,
)


def _conversation_agent(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.model = "gpt-6.1-sol"
    agent.provider = "openai-codex"
    agent.codex_responses_native_compaction = True
    agent.compression_checkpoint_required = False
    agent.runtime_capabilities = {"native_compaction": True}
    agent.capabilities = {}
    agent.context_compressor = _agent().context_compressor
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "Offline native-first test."
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0)
    return agent


@pytest.mark.parametrize("prompt,completion,costs,pressure", HISTORICAL_PRESSURES)
def test_historical_pressure_actual_turn_loop_request_order(monkeypatch, prompt, completion, costs, pressure):
    from agent.usage_anchor import anchored_context_tokens

    agent = _conversation_agent(monkeypatch)
    response = _codex_tool_call_response()
    response.usage = SimpleNamespace(input_tokens=prompt, output_tokens=completion, total_tokens=prompt + completion)
    responses = [response, _codex_message_response("offline fixture answer")]
    events = []
    def request(kwargs):
        events.append(("request", deepcopy(kwargs.get("context_management"))))
        return responses.pop(0)
    def tools(_assistant, messages, *_a, **_kw):
        for i, cost in enumerate(costs):
            message = _tool_at_cost(cost, i)
            message["tool_call_id"] = "call_1"
            messages.append(message)
        assert anchored_context_tokens(messages, agent._usage_anchor) == pressure
        events.append(("tools", pressure))
    agent._compress_context = MagicMock(side_effect=AssertionError("local summarizer preceded native request"))
    monkeypatch.setattr(agent, "_interruptible_api_call", request)
    monkeypatch.setattr(agent, "_execute_tool_calls", tools)
    result = agent.run_conversation("continue")
    assert result["completed"] is True
    assert [e[0] for e in events] == ["request", "tools", "request"]
    assert all(e[1] == [{"type": "compaction", "compact_threshold": 246808}] for e in events if e[0] == "request")
    agent._compress_context.assert_not_called()


@pytest.mark.parametrize("error_text,status", [
    ("Request entity too large", 413),
    ("maximum context length exceeded", 400),
    ("Unknown parameter: context_management", 400),
])
def test_native_request_error_restores_local_recovery(monkeypatch, error_text, status):
    agent = _conversation_agent(monkeypatch)
    agent.context_compressor.context_length = 40000
    agent.context_compressor.threshold_tokens = 20000
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 15000} for i in range(8)]
    error = Exception(error_text)
    error.status_code = status
    responses = [error, _codex_message_response("recovered fixture")]
    events = []
    def request(kwargs):
        events.append(("request", "context_management" in kwargs))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
    def compact(messages, system, **kwargs):
        events.append(("local", kwargs))
        return list(messages[-2:]), system
    monkeypatch.setattr(agent, "_interruptible_api_call", request)
    monkeypatch.setattr(agent, "_compress_context", compact)
    result = agent.run_conversation("continue", conversation_history=history)
    assert result["completed"] is True
    assert result["final_response"] == "recovered fixture"
    assert events[0] == ("request", True)
    assert events[-1] == ("request", "context_management" not in error_text)
    if "context_management" in error_text:
        # Rejection keeps the existing immediate retry without the unsupported
        # parameter. It is not itself a proven context overflow.
        assert [e[0] for e in events] == ["request", "request"]
        assert agent.codex_responses_native_compaction is False
        # A subsequent anchored proactive gate in this same session must now
        # admit local compression instead of retaining a cached native veto.
        from tests.agent.test_native_first_automatic_compaction import _pre_api
        agent._request_pressure_anchored = True
        verdict = _pre_api(agent, result["messages"], 30000)
        assert verdict.action == "continue"
        assert events[-1][0] == "local"
    else:
        assert [e[0] for e in events] == ["request", "local", "request"]
        assert events[1][1]["bypass_cooldown"] is True


@pytest.mark.parametrize("model,provider,url,capabilities", [
    ("gpt-6.1-sol", "openai-codex", "https://chatgpt.com/backend-api/codex", {}),
    ("gpt-6.1-sol-900k", "openai-codex", "https://chatgpt.com/backend-api/codex", {}),
    ("gpt-6-astra", "openai-codex", "https://chatgpt.com/backend-api/codex", {}),
    ("gpt-5.6", "openai", "https://api.openai.com/v1", {}),
    ("gpt-5.6", "custom", "https://trusted.example/v1", {"openai_native_compaction": True}),
])
def test_ownership_tracks_real_wire_gate_and_live_optout(model, provider, url, capabilities):
    from agent.codex_responses_adapter import native_responses_owns_automatic_compaction

    agent = _agent()
    agent.model, agent.provider, agent.base_url, agent.capabilities = model, provider, url, capabilities
    assert native_responses_owns_automatic_compaction(agent) is True
    agent.codex_responses_native_compaction = False
    assert native_responses_owns_automatic_compaction(agent) is False


def test_native_checkpoint_capture_db_reopen_and_fresh_agent_replay(monkeypatch, tmp_path):
    from hermes_state import SessionDB

    agent = _conversation_agent(monkeypatch)
    response = _codex_message_response("checkpoint fixture")
    response.output.insert(0, SimpleNamespace(type="compaction", encrypted_content="synthetic-opaque-checkpoint"))
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda kwargs: response)
    result = agent.run_conversation("remember fixture")
    assert result["completed"] is True
    captured = [m for m in result["messages"] if m.get("codex_reasoning_items")]
    assert any(item["type"] == "compaction" for m in captured for item in m["codex_reasoning_items"])
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("native-reload", source="test")
    for message in result["messages"]:
        db.append_message("native-reload", message["role"], message.get("content"), codex_reasoning_items=message.get("codex_reasoning_items"))
    db.close()
    db = SessionDB(db_path=path)
    history = db.get_messages_as_conversation("native-reload")
    db.close()
    fresh = _conversation_agent(monkeypatch)
    fresh.compression_idle_compact_after_seconds = 1
    fresh._last_activity_ts = 0
    # Inflate only the estimator, never checkpoint provenance/admission/hydration.
    monkeypatch.setattr("agent.codex_responses_adapter.estimate_native_responses_preflight_tokens", lambda *a, **kw: 1300000)
    fresh._compress_context = MagicMock(side_effect=AssertionError("restored checkpoint locally summarized"))
    requests = []
    def request(kwargs):
        requests.append(deepcopy(kwargs))
        return _codex_message_response("resumed fixture")
    monkeypatch.setattr(fresh, "_interruptible_api_call", request)
    resumed = fresh.run_conversation("continue", conversation_history=history)
    assert resumed["completed"] is True
    assert len(requests) == 1
    assert any(item.get("type") == "compaction" and item.get("encrypted_content") == "synthetic-opaque-checkpoint" for item in requests[0]["input"])
    fresh._compress_context.assert_not_called()
    assert fresh.context_compressor.awaiting_real_usage_after_compression is False


def test_native_eligible_manual_compression_preserves_commit_cancellation(monkeypatch, tmp_path):
    from hermes_state import SessionDB
    from agent.conversation_compression import CompressionCommitFence
    from tests.agent.test_compression_concurrent_fork import _build_agent_with_db
    from agent.codex_responses_adapter import native_responses_owns_automatic_compaction

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("native-manual", source="test")
    agent = _build_agent_with_db(db, "native-manual")
    agent.model, agent.provider, agent.api_mode = "gpt-6.1-sol", "openai-codex", "codex_responses"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.compression_enabled = agent.codex_responses_native_compaction = True
    agent.runtime_capabilities = {"native_compaction": True}
    agent.compression_checkpoint_required = False
    agent.context_compressor.threshold_tokens = 255000
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    assert native_responses_owns_automatic_compaction(agent) is True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    fence = CompressionCommitFence()
    summary = [{"role": "user", "content": "[CONTEXT COMPACTION] fixture summary"}, {"role": "user", "content": "tail"}]
    def cancel_before_return(*a, **kw):
        assert fence.cancel_before_commit() is True
        return deepcopy(summary)
    agent.context_compressor.compress.side_effect = cancel_before_return
    before = db.get_messages("native-manual")
    cancelled, _ = agent._compress_context(messages, "sys", force=True, commit_fence=fence)
    assert cancelled is messages
    assert db.get_messages("native-manual") == before
    assert db.get_compression_lock_holder("native-manual") is None
    agent.context_compressor.compress.side_effect = lambda *a, **kw: deepcopy(summary)
    manual, _ = agent._compress_context(messages, "sys", force=True)
    assert len(manual) == 2 and manual is not messages
    assert agent._last_compaction_in_place is True
    assert db.get_compression_lock_holder("native-manual") is None
    assert len(db.get_messages("native-manual")) == 2
    db.close()
