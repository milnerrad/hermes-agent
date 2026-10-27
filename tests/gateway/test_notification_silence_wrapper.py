"""Quiet notification exits survive the real runner, shaping, persistence and adapter egress.

Only model execution and the platform transport are scripted; neither the runner's result
projection nor the gateway's final-response decisions are replaced.
"""
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.response_filters import display_kind_for_event
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from tests.gateway.test_gateway_silence_tokens import _event, _runner


class RecordingPlatform(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id=str(len(self.sent)))

    async def get_chat_info(self, chat_id):
        return {"name": "test", "type": "group"}


class ScriptedAgent:
    def __init__(self, result, **kwargs):
        self.result = result
        self.model = kwargs["model"]
        self.provider = "scripted"
        self.session_id = kwargs["session_id"]
        self.tools = []
        self.context_compressor = SimpleNamespace(
            last_prompt_tokens=100, context_length=200_000, compression_count=0)
        self.session_prompt_tokens = 120
        self.session_completion_tokens = 5
        self.calls = []

    def run_conversation(self, message, **kwargs):
        self.calls.append((message, kwargs))
        self._session_messages = [*kwargs["conversation_history"],
            {"role": "user", "content": message,
             **({"display_kind": kwargs["persist_user_display_kind"]}
                if kwargs.get("persist_user_display_kind") else {})},
            {"role": "assistant", "content": self.result["final_response"] or "[SILENT]"}]
        return {"messages": self._session_messages, **self.result}


async def _deliver(monkeypatch, tmp_path, result, *, internal=True, kind=None, reply_expected=None,
                   history=None, agent=None):
    runner = _runner(monkeypatch, tmp_path)
    platform = RecordingPlatform()
    event = _event(internal=internal, reply_expected=reply_expected)
    if kind:
        event.metadata.update(display_kind=kind, display_text="Task finished")
    runner.adapters = {Platform.TELEGRAM: platform}
    runner._delivery_adapter_for = lambda _source: platform
    runner._resolve_session_agent_runtime = lambda **_kw: ("scripted/model", {"api_key": "fake"})
    runner._resolve_session_reasoning_config = lambda **_kw: None
    runner._resolve_session_service_tier = lambda **_kw: None
    runner._refresh_fallback_model = lambda: None
    runner._resolve_turn_agent_config = lambda _msg, model, runtime: {"model": model, "runtime": runtime}
    runner._consume_pending_native_image_paths = lambda _key: []
    runner._consume_pending_turn_sidecar_notes = lambda _key: []
    runner._get_system_prompt_for_channel = lambda *_a, **_kw: None
    runner.config.streaming = None
    runner._session_db = None
    entry = runner.session_store.get_or_create_session.return_value
    runner.session_store._entries = {entry.session_key: entry}
    runner.session_store.has_input_owner.return_value = False
    runner.session_store.load_transcript.return_value = history or []
    contexts, projected = [], []

    def build_agent(**kwargs):
        if agent is not None:
            agent.session_id = kwargs["session_id"]
            return agent
        return ScriptedAgent(result, **kwargs)

    async def run_agent(**kwargs):
        kwargs.pop("message_type", None)
        ctx = TurnContext(**kwargs, user_config={},
            AIAgent=build_agent,
            resolve_display_setting=lambda *_a: False, _run_still_current=lambda: True,
            _hooks_ref=SimpleNamespace(loaded_hooks=False))
        contexts.append(ctx)
        value = TurnRunner(runner, ctx).run_sync()
        projected.append(value)
        return value

    runner._run_agent = run_agent

    async def handle(inbound):
        return await runner._handle_message_with_agent(inbound, inbound.source, entry.session_key, 1)

    platform.set_message_handler(handle)
    await platform._process_message_background(event, entry.session_key)
    assert len(projected) == 1, "the real runner must execute before delivery"
    return platform, runner, contexts[0], projected[0], event


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "process_complete", "async_delegation_complete"])
async def test_blank_notification_exit_reaches_platform_without_warning(monkeypatch, tmp_path, kind):
    result = {"final_response": "", "failed": False, "completed": True,
              "turn_exit_reason": "notification_no_reply", "api_calls": 3,
              "cache_read_tokens": 90, "cache_write_tokens": 10}
    history = [{"role": "user", "content": "Check the task"},
               {"role": "assistant", "content": "", "tool_calls": [
                   {"id": "old-call", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
               {"role": "tool", "content": "done", "tool_call_id": "old-call"},
               {"role": "assistant", "content": "Waiting for completion."}]
    platform, runner, ctx, projected, event = await _deliver(
        monkeypatch, tmp_path, result, kind=kind, history=history)

    assert platform.sent == []
    assert projected["history_offset"] == len(history)
    assert projected["messages"][:len(history)] == history
    assert runner._agent_cache[ctx.session_key][0] is ctx.agent_holder[0]
    assert projected["final_response"] == ""
    assert projected["api_calls"] == 3
    assert projected["cache_read_tokens"] == 90
    assert projected["cache_write_tokens"] == 10
    assert projected["messages"] == ctx.agent_holder[0]._session_messages
    assert ctx.agent_holder[0].calls[0][1]["persist_user_display_kind"] == display_kind_for_event(event)
    appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert any(row.get("role") == "user" and row.get("display_kind") == display_kind_for_event(event)
               for row in appended)
    assert any(row.get("role") == "assistant" and row.get("content") == "[SILENT]" for row in appended)


@pytest.mark.asyncio
@pytest.mark.parametrize("internal,reply_expected,extra", [
    (True, None, {"turn_exit_reason": None}),
    (True, None, {"failed": True, "error": "provider unavailable"}),
    (True, None, {"partial": True, "error": "output truncated"}),
    (True, None, {"completed": False}),
    (True, None, {"interrupted": True, "api_calls": 0}),
    (True, None, {"error": "unfinished provider response"}),
    (True, None, {"failure_reason": "incomplete"}),
    (True, None, {"compression_deferred": True}),
    (True, None, {"compression_exhausted": True}),
    (True, None, {"partial": True, "completed": False, "error": "mixed incomplete causes"}),
    (False, None, {}),
    (False, True, {}),
    (False, False, {}),
], ids=["ordinary-empty", "failed", "partial", "unfinished", "interrupted-before-start",
        "error-with-quiet-reason", "failure-with-quiet-reason", "pending-compression",
        "exhausted-compression", "coexisting-causes", "human", "addressed-human", "unaddressed-human"])
async def test_invalid_empty_results_remain_visible(monkeypatch, tmp_path, internal, reply_expected, extra):
    result = {"final_response": "", "failed": False, "completed": True,
              "turn_exit_reason": "notification_no_reply", "api_calls": 1, **extra}
    platform, _runner, _ctx, projected, _event = await _deliver(
        monkeypatch, tmp_path, result, internal=internal, reply_expected=reply_expected)
    assert platform.sent and all(text.strip() for text in platform.sent)
    assert projected["api_calls"] == result["api_calls"]
    for key in ("failed", "partial", "completed", "interrupted", "error", "failure_reason"):
        if key in result:
            assert projected[key] == result[key]


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [
    [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call-pending", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]}],
    [{"role": "tool", "content": "still needs interpretation", "tool_call_id": "call-pending"}],
])
async def test_pending_tool_work_cannot_be_hidden_by_quiet_exit(monkeypatch, tmp_path, tail):
    result = {"final_response": "", "failed": False, "completed": True,
              "turn_exit_reason": "notification_no_reply", "api_calls": 2,
              "messages": [{"role": "user", "content": "notice", "display_kind": "internal_notification"}, *tail]}
    platform, _runner, _ctx, projected, _event = await _deliver(monkeypatch, tmp_path, result)
    assert platform.sent and all(text.strip() for text in platform.sent)
    assert projected["messages"] == result["messages"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["The build passed; the artifact is ready.", "Use NO_REPLY only when nothing needs attention."])
async def test_substantive_notification_is_delivered_even_with_quiet_reason(monkeypatch, tmp_path, text):
    result = {"final_response": text, "failed": False, "completed": True,
              "turn_exit_reason": "notification_no_reply", "api_calls": 2}
    platform, _runner, _ctx, _projected, _event = await _deliver(monkeypatch, tmp_path, result)
    assert platform.sent == [text]


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{"partial": True}, {"interrupted": True}, {"completed": False}, {"error": "unfinished"}])
async def test_unsuccessful_silence_marker_does_not_suppress_delivery(monkeypatch, tmp_path, extra):
    result = {"final_response": "[SILENT]", "failed": False, "completed": True,
              "turn_exit_reason": "notification_no_reply", "api_calls": 1, **extra}
    platform, _runner, _ctx, _projected, _event = await _deliver(monkeypatch, tmp_path, result)
    assert platform.sent and all(text.strip() for text in platform.sent)


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["quiet-reasoning", "unfinished-child", "leaked-action", "pending-child"])
async def test_codex_provenance_guard_survives_runner_and_delivery(monkeypatch, tmp_path, cause):
    from tests.agent.test_notification_silence_recovery_causes import drive
    from tests.agent.test_run_agent_codex_responses import _codex_message_response, _codex_reasoning_only_response

    first = _codex_reasoning_only_response()
    if cause != "quiet-reasoning":
        child = _codex_message_response('to=functions.terminal {"command":"pwd"}' if cause == "leaked-action" else "").output[0]
        if cause in {"unfinished-child", "pending-child"}:
            child.status = "incomplete" if cause == "unfinished-child" else "in_progress"
        first.output.append(child)
    responses = [first] if cause == "quiet-reasoning" else [first, _codex_message_response("Verified new result.")]
    agent, api = drive(monkeypatch, responses)
    platform, runner, ctx, projected, _event = await _deliver(monkeypatch, tmp_path, None, agent=agent)
    assert ctx.agent_holder[0] is agent
    assert runner._agent_cache[ctx.session_key][0] is agent
    assert projected["api_calls"] == api.call_count == len(responses)
    if cause == "quiet-reasoning":
        assert projected["final_response"] == ""
        assert projected["turn_exit_reason"] == "notification_no_reply"
        assert platform.sent == []
    else:
        assert projected["turn_exit_reason"] != "notification_no_reply"
        assert platform.sent == ["Verified new result."]
