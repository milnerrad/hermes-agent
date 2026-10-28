"""Offline dispatch regressions: native owns proactive work, not explicit recovery.

Historical fixtures preserve the measured provider counts and tool-message costs from
session 20261027_212643_097e8268, not private tool contents. Payloads are synthetic
cost-equivalent stand-ins; no production database or model endpoint is accessed.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_messages_tokens_rough
from agent.native_compaction import native_compaction_context_management
from agent.turn_context_compaction import CompactionOutcome, _idle_compaction, _preflight_compression
from agent.turn_preflight import compress_after_tool_results
from agent.turn_preflight_gate import run_preflight_gate
from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor


# prompt, completion, measured appended tool costs, measured host pressure
HISTORICAL_PRESSURES = [
    pytest.param(253087, 3037, [44], 256168, id="23h05-18-char-tool"),
    pytest.param(254955, 1278, [609, 557], 257399, id="11h15-two-tools"),
]


def _tool_at_cost(cost, index=0):
    msg = {"role": "tool", "tool_call_id": f"call_{index}", "content": "", "_db_persisted": True}
    msg["content"] = "x" * ((cost - estimate_messages_tokens_rough([msg])) * 4)
    assert estimate_messages_tokens_rough([msg]) == cost
    return msg


def _agent():
    compressor = ContextCompressor(
        model="gpt-6.1-sol", threshold_percent=.85, config_context_length=300000,
        provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses", quiet_mode=True,
    )
    agent = SimpleNamespace(
        model="gpt-6.1-sol", provider="openai-codex", api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex", compression_enabled=True,
        codex_responses_native_compaction=True, compression_checkpoint_required=False,
        runtime_capabilities={"native_compaction": True}, capabilities={}, tools=[],
        context_compressor=compressor, _compression_feasibility_checked=True,
        session_id="offline-native-first", _request_pressure_anchored=True,
        _emit_status=MagicMock(), _safe_print=MagicMock(),
        _warn_context_overflow_blocked=MagicMock(), iteration_budget=MagicMock(),
        compression_idle_compact_after_seconds=1, _last_activity_ts=0,
    )
    agent._compress_context = MagicMock(side_effect=lambda messages, system, **kw: (messages, system))
    return agent


def _post_tool(agent, messages):
    return compress_after_tool_results(
        agent, messages=messages, system_message="sys", user_message="continue",
        active_system_prompt="sys", conversation_history=messages[:1],
        compression_attempts=0, max_compression_attempts=3, effective_task_id="t",
        final_response="", turn_exit_reason="", current_turn_user_idx=0,
    )


def _pre_api(agent, messages, pressure, *, recovery=False):
    return run_preflight_gate(
        agent, request_pressure_tokens=pressure, _moa_prepared_request=None,
        pending_moa_prepared_request=None, messages=messages, system_message="sys",
        user_message="continue", active_system_prompt="sys", conversation_history=messages[:1],
        api_call_count=1, compression_attempts=0, max_compression_attempts=3,
        effective_task_id="t", final_response="", failed=False, _turn_exit_reason="",
        _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=recovery, _last_preflight_pressure=None,
    )


@pytest.mark.parametrize("prompt,completion,costs,pressure", HISTORICAL_PRESSURES)
def test_historical_pressure_tracer_reaches_next_native_request_without_local_summary(prompt, completion, costs, pressure):
    agent = _agent()
    base = [{"role": "user", "content": "continue"}]
    agent._usage_anchor = capture_usage_anchor(prompt, completion, base)
    agent.context_compressor.last_prompt_tokens = prompt
    messages = base + [{"role": "assistant", "content": ""}] + [
        _tool_at_cost(cost, i) for i, cost in enumerate(costs)
    ]
    original = deepcopy(messages)
    assert anchored_context_tokens(messages, agent._usage_anchor) == pressure
    assert agent.context_compressor.threshold_tokens == 255000
    assert native_compaction_context_management(agent, is_codex_backend=True) == [
        {"type": "compaction", "compact_threshold": 246808}
    ]
    post = _post_tool(agent, messages)
    pre = _pre_api(agent, post.messages, pressure)
    # This tracer records request admission, not a fabricated provider success.
    assert pre.action == "fallthrough"
    assert messages == original and pre.messages is messages
    assert post.compression_attempts == pre.compression_attempts == 0
    agent._compress_context.assert_not_called()
    agent._warn_context_overflow_blocked.assert_not_called()


@pytest.mark.parametrize("phase", ["turn_start", "idle", "pre_api", "post_tool"])
def test_native_owns_every_automatic_entry_point(monkeypatch, phase):
    agent = _agent()
    _run_phase(monkeypatch, phase, agent)
    agent._compress_context.assert_not_called()
    agent._warn_context_overflow_blocked.assert_not_called()


def _run_phase(monkeypatch, phase, agent):
    messages = [{"role": "user", "content": "continue"}, {"role": "assistant", "content": "history"}]
    if phase == "pre_api":
        return _pre_api(agent, messages, 260000)
    if phase == "post_tool":
        agent.context_compressor.last_prompt_tokens = 260000
        return _post_tool(agent, messages)
    monkeypatch.setattr("agent.turn_context._should_run_preflight_estimate", lambda *a: True)
    monkeypatch.setattr("agent.turn_context._preflight_request_tokens", lambda *a: 260000)
    out = CompactionOutcome(messages, "sys", messages[:1], 0)
    fn = _idle_compaction if phase == "idle" else _preflight_compression
    fn(agent, out, "sys", "continue", "t")
    return out


@pytest.mark.parametrize("phase", ["turn_start", "idle", "pre_api", "post_tool"])
@pytest.mark.parametrize("change", [
    {"codex_responses_native_compaction": False},
    {"model": "gpt-5.2"},
    {"model": "gpt-6.1-sol-mini"},
    {"compression_checkpoint_required": True},
    {"runtime_capabilities": {"native_compaction": False}},
    {"api_mode": "chat_completions"},
    {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
    {"provider": "xai", "model": "gpt-5.6", "base_url": "https://api.x.ai/v1"},
    {"provider": "copilot", "model": "gpt-5.6", "base_url": "https://models.github.ai/inference"},
])
def test_ineligible_request_preserves_local_automatic_path(monkeypatch, phase, change):
    agent = _agent()
    vars(agent).update(change)
    _run_phase(monkeypatch, phase, agent)
    agent._compress_context.assert_called_once()


def test_provider_proven_overflow_preflight_keeps_local_recovery():
    agent = _agent()
    messages = [{"role": "user", "content": "continue"}, {"role": "assistant", "content": "history"}]
    verdict = _pre_api(agent, messages, 260000, recovery=True)
    agent._compress_context.assert_called_once()
    assert agent._compress_context.call_args.kwargs["trigger"] == "pre_api"
    assert verdict.action == "continue"
    assert verdict.compression_attempts == 1


@pytest.mark.parametrize("phase", ["turn_start", "idle", "pre_api", "post_tool"])
def test_disabled_compression_neither_native_nor_local(monkeypatch, phase):
    agent = _agent()
    agent.compression_enabled = False
    _run_phase(monkeypatch, phase, agent)
    agent._compress_context.assert_not_called()


@pytest.mark.parametrize("phase", ["turn_start", "idle", "pre_api", "post_tool"])
def test_native_admission_precedes_auxiliary_feasibility_and_destructive_maintenance(monkeypatch, phase):
    agent = _agent()
    agent._compression_feasibility_checked = False
    def forbidden(*a, **kw):
        raise AssertionError("local feasibility/pruning ran before native admission")
    for module in ("agent.turn_preflight", "agent.conversation_compression"):
        monkeypatch.setattr(module + ".ensure_compression_feasibility_checked", forbidden)
    monkeypatch.setattr(agent.context_compressor, "prune_tool_results_only", forbidden)
    monkeypatch.setattr(agent.context_compressor, "snapshot_preflight_display_tokens", forbidden)
    monkeypatch.setattr(agent.context_compressor, "should_compress_info", forbidden)
    _run_phase(monkeypatch, phase, agent)
    agent._compress_context.assert_not_called()
    assert agent.context_compressor.threshold_tokens == 255000


def test_large_tool_jump_cannot_steal_native_first_request():
    agent = _agent()
    base = [{"role": "user", "content": "continue"}]
    agent._usage_anchor = capture_usage_anchor(240000, 10000, base)
    messages = base + [{"role": "assistant", "content": ""}, _tool_at_cost(12000)]
    assert anchored_context_tokens(messages, agent._usage_anchor) == 262000
    original = deepcopy(messages)
    post = _post_tool(agent, messages)
    pre = _pre_api(agent, post.messages, 262000)
    assert pre.action == "fallthrough"
    assert messages == original
    agent._compress_context.assert_not_called()
