"""Reviewer P1 regressions at the real mocked-provider and gateway loop boundaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from tests.agent.test_run_agent_codex_responses import (
    _build_agent,
    _codex_message_response,
    _codex_reasoning_only_response,
    _codex_tool_call_response,
)
from tests.gateway.test_gateway_silence_tokens import _runner, _event, _source

KINDS = ("internal_notification", "process_complete", "async_delegation_complete")


def drive(monkeypatch, responses):
    agent = _build_agent(monkeypatch)
    agent.compression_enabled = False
    agent._cached_system_prompt = "You are helpful."
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 0)
    api = MagicMock(side_effect=responses)
    monkeypatch.setattr(agent, "_interruptible_api_call", api)
    return agent, api


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "leak",
    [
        'to=functions.terminal {"command":"pwd"}',
        'Running the verification now.\n{"cmd":"pwd"}',
    ],
)
def test_leaked_action_still_recovers(monkeypatch, kind, leak):
    agent, api = drive(
        monkeypatch,
        [
            _codex_message_response(leak),
            _codex_message_response("Recovered actionable answer."),
        ],
    )
    result = agent.run_conversation(
        "Verify the completed task before reporting success.",
        persist_user_display_kind=kind,
    )
    assert (api.call_count, result["final_response"], result["turn_exit_reason"]) != (
        1,
        "",
        "notification_no_reply",
    ), result
    assert api.call_count == 2
    assert result["final_response"] == "Recovered actionable answer."


@pytest.mark.parametrize("kind", KINDS)
def test_incomplete_child_item_still_recovers(monkeypatch, kind):
    first = _codex_message_response("")
    first.output[0].status = "incomplete"
    agent, api = drive(
        monkeypatch, [first, _codex_message_response("Recovered incomplete item.")]
    )
    result = agent.run_conversation(
        "Verify background completion.", persist_user_display_kind=kind
    )
    assert api.call_count == 2, result
    assert result["final_response"] == "Recovered incomplete item."


@pytest.mark.parametrize(
    "internal, failed, text, quiet",
    [
        (True, False, "", True),
        (False, False, "", False),
        (True, True, "", False),
        (True, False, "Action required: verification failed.", False),
        (False, False, "Human reply.", False),
    ],
)
@pytest.mark.asyncio
async def test_actual_gateway_new_exit_boundaries(
    monkeypatch, tmp_path, internal, failed, text, quiet
):
    runner = _runner(monkeypatch, tmp_path)
    result = {
        "final_response": text,
        "turn_exit_reason": "notification_no_reply",
        "failed": failed,
        "messages": [
            {"role": "user", "content": "notice"},
            {"role": "assistant", "content": text or "[SILENT]"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
    }
    if failed:
        result["error"] = "Independent test provider failure; user action required."
    runner._run_agent = AsyncMock(return_value=result)
    response = await runner._handle_message_with_agent(
        _event(internal=internal), _source(), "agent:main:telegram:group:-1001:12345", 1
    )
    assert (response == "") is quiet, (result, response)
    if text:
        assert response == text


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "leak",
    [
        'to=functions.terminal {"command":"pwd"}',
        'Running the verification now.\n{"cmd":"pwd"}',
    ],
)
def test_reasoning_with_leaked_action_still_recovers(monkeypatch, kind, leak):
    first = _codex_reasoning_only_response()
    first.output.extend(_codex_message_response(leak).output)
    agent, api = drive(
        monkeypatch, [first, _codex_message_response("Recovered mixed action.")]
    )
    result = agent.run_conversation(
        "Verify the late result.", persist_user_display_kind=kind
    )
    assert api.call_count == 2, result
    assert result["final_response"] == "Recovered mixed action."
    assert result.get("turn_exit_reason") != "notification_no_reply"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("status", ["incomplete", "queued", "in_progress"])
@pytest.mark.parametrize("child", ["message", "function_call", "reasoning"])
def test_reasoning_with_unfinished_child_still_recovers(
    monkeypatch, kind, status, child
):
    first = _codex_reasoning_only_response()
    if child == "reasoning":
        first.output[0].status = status
    else:
        output = (
            _codex_message_response("")
            if child == "message"
            else _codex_tool_call_response()
        ).output
        output[0].status = status
        first.output.extend(output)
    agent, api = drive(
        monkeypatch, [first, _codex_message_response("Recovered unfinished child.")]
    )
    tool = MagicMock(
        side_effect=AssertionError("Truncated function call must not execute.")
    )
    monkeypatch.setattr("model_tools.handle_function_call", tool)
    result = agent.run_conversation(
        "Verify the late result.", persist_user_display_kind=kind
    )
    assert api.call_count == 2, result
    assert result["final_response"] == "Recovered unfinished child."
    assert result.get("turn_exit_reason") != "notification_no_reply"
    tool.assert_not_called()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("status", ["incomplete", "queued", "in_progress", None])
def test_reasoning_without_completed_response_still_recovers(monkeypatch, kind, status):
    first = _codex_reasoning_only_response()
    first.status = status
    agent, api = drive(
        monkeypatch, [first, _codex_message_response("Recovered provider state.")]
    )
    result = agent.run_conversation(
        "Verify the late result.", persist_user_display_kind=kind
    )
    assert api.call_count == 2, result
    assert result["final_response"] == "Recovered provider state."


@pytest.mark.parametrize("kind", KINDS)
def test_reasoning_with_commentary_not_proven_final_still_recovers(monkeypatch, kind):
    first = _codex_reasoning_only_response()
    output = _codex_message_response("Running verification now.").output
    output[0].phase = "commentary"
    first.output.extend(output)
    agent, api = drive(
        monkeypatch, [first, _codex_message_response("Verified final answer.")]
    )
    result = agent.run_conversation(
        "Verify the late result.", persist_user_display_kind=kind
    )
    assert api.call_count == 2, result
    assert result["final_response"] == "Verified final answer."


@pytest.mark.parametrize("kind", KINDS)
def test_legacy_normalized_message_without_reason_metadata_fails_closed(
    monkeypatch, kind
):
    from agent.turn_truncation import normalize_response_for_agent

    first = _codex_reasoning_only_response()

    def legacy(agent, response):
        if response is first:
            return SimpleNamespace(
                content="",
                tool_calls=None,
                finish_reason="incomplete",
                reasoning="Thinking...",
                reasoning_content=None,
                reasoning_details=None,
                codex_reasoning_items=None,
                codex_message_items=None,
            )
        return normalize_response_for_agent(agent, response)

    monkeypatch.setattr(
        "agent.turn_response_intake.normalize_response_for_agent", legacy
    )
    agent, api = drive(
        monkeypatch, [first, _codex_message_response("Recovered legacy normalization.")]
    )
    result = agent.run_conversation(
        "Verify the late result.", persist_user_display_kind=kind
    )
    assert api.call_count == 2, result
    assert result["final_response"] == "Recovered legacy normalization."


@pytest.mark.parametrize("kind", KINDS)
def test_codex_reasoning_only_notice_preserves_streamed_partial(monkeypatch, kind):
    agent, api = drive(monkeypatch, [])

    def streamed(*args, **kwargs):
        agent._current_streamed_assistant_text = "Correction: late failure."
        return _codex_reasoning_only_response()

    api.side_effect = streamed
    result = agent.run_conversation("Late result", persist_user_display_kind=kind)
    assert api.call_count == 1
    assert result["final_response"].startswith("Correction: late failure.")
    assert result.get("turn_exit_reason") != "notification_no_reply"
