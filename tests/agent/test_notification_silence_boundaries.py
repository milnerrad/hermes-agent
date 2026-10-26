"""Installed-base boundary coverage beyond upstream #139767; no real provider calls."""
from unittest.mock import MagicMock

import pytest

from gateway.response_filters import is_intentional_silence_agent_result
from tests.agent.test_notification_turn_empty_reply import (
    _completed_tool_round,
    _empty_response,
    _text_response,
    loop_agent,
)
from tests.agent.test_run_agent_codex_responses import (
    _build_agent,
    _codex_message_response,
    _codex_reasoning_only_response,
    _codex_tool_call_response,
)

KINDS = ("internal_notification", "process_complete", "async_delegation_complete")


@pytest.mark.parametrize("kind", KINDS)
def test_all_machinery_kinds_quiet_once_without_fallback(loop_agent, monkeypatch, kind):
    fallback = MagicMock(return_value=False)
    monkeypatch.setattr(loop_agent, "_try_activate_fallback", fallback)
    result = loop_agent.run_conversation(
        "Late result: tests still pass.", conversation_history=_completed_tool_round(),
        persist_user_display_kind=kind,
    )
    assert result["api_calls"] == 1
    assert result["final_response"] == ""
    assert result["turn_exit_reason"] == "notification_no_reply"
    assert result["completed"] is True
    assert not result.get("failed")
    assert result["messages"][-1]["content"] == "[SILENT]"
    assert is_intentional_silence_agent_result(result, result["final_response"])
    fallback.assert_not_called()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("text", (
    "Correction: one test failed; do not deploy this candidate.",
    "The background task failed with permission denied. User action is required.",
))
def test_substantive_notice_answer_is_never_suppressed(loop_agent, kind, text):
    loop_agent.client.chat.completions.create.side_effect = lambda **kw: _text_response(text)
    result = loop_agent.run_conversation("New background result", persist_user_display_kind=kind)
    assert result["final_response"] == text
    assert result["api_calls"] == 1
    assert not is_intentional_silence_agent_result(result, text)


def _drive_codex(monkeypatch, responses):
    agent = _build_agent(monkeypatch)
    agent.compression_enabled = False
    agent._cached_system_prompt = "You are helpful."
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0)
    api = MagicMock(side_effect=responses)
    monkeypatch.setattr(agent, "_interruptible_api_call", api)
    return agent, api


@pytest.mark.parametrize("kind", KINDS)
def test_codex_completed_reasoning_only_notice_bypasses_pre_nudge(monkeypatch, kind):
    agent, api = _drive_codex(monkeypatch, [
        _codex_reasoning_only_response(), _codex_message_response("Redundant acknowledgement."),
    ])
    fallback = MagicMock(return_value=False)
    monkeypatch.setattr(agent, "_try_activate_fallback", fallback)
    result = agent.run_conversation("Tests still pass.", persist_user_display_kind=kind)
    assert api.call_count == 1
    assert result["final_response"] == ""
    assert result["turn_exit_reason"] == "notification_no_reply"
    assert result["messages"][-1]["content"] == "[SILENT]"
    assert not any("_codex_incomplete" in str(row) for row in result["messages"])
    fallback.assert_not_called()


def test_codex_human_reasoning_only_keeps_continuation(monkeypatch):
    agent, api = _drive_codex(monkeypatch, [
        _codex_reasoning_only_response(), _codex_message_response("Human answer."),
    ])
    result = agent.run_conversation("Please explain the results.")
    assert api.call_count == 2
    assert result["final_response"] == "Human answer."
    assert not is_intentional_silence_agent_result(result, result["final_response"])


@pytest.mark.parametrize("kind", KINDS)
def test_codex_truly_incomplete_notice_still_recovers(monkeypatch, kind):
    response = _codex_reasoning_only_response()
    response.status = "incomplete"
    response.incomplete_details = {"reason": "max_output_tokens"}
    agent, api = _drive_codex(monkeypatch, [response, _codex_message_response("Correction: failed.")])
    result = agent.run_conversation("Task completed", persist_user_display_kind=kind)
    assert api.call_count == 2
    assert result["final_response"] == "Correction: failed."
    assert not is_intentional_silence_agent_result(result, result["final_response"])


@pytest.mark.parametrize("kind", KINDS)
def test_notice_that_takes_tool_action_keeps_recovery(monkeypatch, kind):
    agent, api = _drive_codex(monkeypatch, [
        _codex_tool_call_response(), _codex_reasoning_only_response(),
        _codex_message_response("The check failed. Stop deployment."),
    ])
    tool = MagicMock(return_value='{"output":"check failed", "exit_code":1}')
    monkeypatch.setattr("model_tools.handle_function_call", tool)
    result = agent.run_conversation("Verify the late result", persist_user_display_kind=kind)
    assert api.call_count == 3
    tool.assert_called_once()
    assert result["final_response"] == "The check failed. Stop deployment."
    assert not is_intentional_silence_agent_result(result, result["final_response"])


def test_streamed_partial_wins_over_empty_notice(loop_agent):
    # The provider boundary may have streamed useful text despite an empty returned message.
    def partial(**kwargs):
        loop_agent._current_streamed_assistant_text = "Correction: late failure."
        return _empty_response()
    loop_agent.client.chat.completions.create.side_effect = partial
    result = loop_agent.run_conversation("Late result", persist_user_display_kind="process_complete")
    assert result["api_calls"] == 1
    assert result["final_response"].startswith("Correction: late failure.")
    assert not is_intentional_silence_agent_result(result, result["final_response"])
