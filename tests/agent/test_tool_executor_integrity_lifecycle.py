"""Lifecycle rejection tests for incomplete historical tool calls."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.run_agent.test_run_agent import agent, _mock_assistant_msg, _mock_tool_call

RESERVED = "__hermes_incomplete_tool_arguments__"


def _run(agent, concurrent, call, messages):
    msg = _mock_assistant_msg(content="", tool_calls=[call])
    method = agent._execute_tool_calls_concurrent if concurrent else agent._execute_tool_calls_sequential
    method(msg, messages, "task-1")


@pytest.mark.parametrize("concurrent", [False, True])
def test_initial_rejection_skips_hooks_and_callbacks(agent, monkeypatch, concurrent):
    events = []
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *args, **kwargs: events.append("hook") or [],
    )
    agent.tool_progress_callback = lambda *args, **kwargs: events.append("progress")
    agent.tool_start_callback = lambda *args, **kwargs: events.append("start")
    agent.tool_complete_callback = lambda *args, **kwargs: events.append("complete")
    agent._touch_activity = lambda *args, **kwargs: events.append("activity")
    payload = {RESERVED: {"version": 1, "replayable": False}}
    call = _mock_tool_call(name="terminal", arguments=json.dumps(payload), call_id="c1")
    messages = []
    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)
    assert events == []
    assert len(messages) == 1
    assert json.loads(messages[0]["content"])["error_type"] == "incomplete_historical_tool_arguments"


@pytest.mark.parametrize("concurrent", [False, True])
def test_deferred_rejection_skips_middleware_hooks_and_callbacks(
    agent, monkeypatch, concurrent
):
    from tools.registry import registry
    from tools.tool_search import TOOL_CALL_NAME

    events = []
    tool_name = "mcp__integrity_lifecycle__run"
    registry.register(
        name=tool_name,
        toolset="integrity-lifecycle",
        schema={
            "name": tool_name,
            "description": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kwargs: events.append("handler") or "handled",
    )
    middleware = lambda **kwargs: events.append("middleware") or None
    manager = SimpleNamespace(
        _middleware={"tool_request": [middleware], "tool_execution": [middleware]}
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *args, **kwargs: events.append("hook") or [],
    )
    agent.tool_progress_callback = lambda *args, **kwargs: events.append("progress")
    agent.tool_start_callback = lambda *args, **kwargs: events.append("start")
    agent.tool_complete_callback = lambda *args, **kwargs: events.append("complete")
    agent._touch_activity = lambda *args, **kwargs: events.append("activity")
    inner = json.dumps({RESERVED: {"version": 1, "replayable": False}})
    outer = json.dumps({"name": tool_name, "arguments": inner})
    call = _mock_tool_call(name=TOOL_CALL_NAME, arguments=outer, call_id="c1")
    messages = []
    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)
    assert events == []
    assert len(messages) == 1
    result = json.loads(messages[0]["content"])
    assert result["error_type"] == "incomplete_historical_tool_arguments"


@pytest.mark.parametrize("concurrent", [False, True])
def test_schema_decoded_rejection_skips_lifecycle(agent, monkeypatch, concurrent):
    from tools.registry import registry

    events = []
    tool_name = "integrity_schema_lifecycle_probe"
    registry.register(
        name=tool_name,
        toolset="integrity-lifecycle",
        schema={
            "name": tool_name,
            "description": "probe",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            },
        },
        handler=lambda args, **kwargs: events.append("handler") or "handled",
    )
    middleware = lambda **kwargs: events.append("middleware") or None
    manager = SimpleNamespace(
        _middleware={"tool_request": [middleware], "tool_execution": [middleware]}
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *args, **kwargs: events.append("hook") or [],
    )
    agent.tool_progress_callback = lambda *args, **kwargs: events.append("progress")
    agent.tool_start_callback = lambda *args, **kwargs: events.append("start")
    agent.tool_complete_callback = lambda *args, **kwargs: events.append("complete")
    agent._touch_activity = lambda *args, **kwargs: events.append("activity")
    encoded = json.dumps([{RESERVED: {"version": 1, "replayable": False}}])
    call = _mock_tool_call(
        name=tool_name,
        arguments=json.dumps({"items": encoded}),
        call_id="c1",
    )
    messages = []
    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)
    assert events == []
    assert json.loads(messages[0]["content"])["error_type"] == "incomplete_historical_tool_arguments"


@pytest.mark.parametrize("concurrent", [False, True])
def test_escaped_schema_decoded_marker_skips_lifecycle(agent, monkeypatch, concurrent):
    from tools.registry import registry

    events = []
    tool_name = "integrity_escaped_schema_lifecycle_probe"
    registry.register(
        name=tool_name,
        toolset="integrity-lifecycle",
        schema={
            "name": tool_name,
            "description": "probe",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            },
        },
        handler=lambda args, **kwargs: events.append("handler") or "handled",
    )
    encoded = '[{"__hermes_incomplete_tool_argument\\u0073__":{"version":1}}]'
    raw_arguments = json.dumps({"items": encoded})
    assert RESERVED not in raw_arguments
    call = _mock_tool_call(name=tool_name, arguments=raw_arguments, call_id="escaped")
    messages = []
    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)
    assert events == []
    assert json.loads(messages[0]["content"])["error_type"] == "incomplete_historical_tool_arguments"


@pytest.mark.parametrize("concurrent", [False, True])
def test_preexisting_interrupt_still_rejects_integrity_before_hooks(
    agent, monkeypatch, concurrent
):
    events = []
    agent._interrupt_requested = True
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *args, **kwargs: events.append("hook") or [],
    )
    agent.tool_progress_callback = lambda *args, **kwargs: events.append("progress")
    agent.tool_start_callback = lambda *args, **kwargs: events.append("start")
    agent.tool_complete_callback = lambda *args, **kwargs: events.append("complete")
    agent._touch_activity = lambda *args, **kwargs: events.append("activity")
    payload = {RESERVED: {"version": 1, "replayable": False}}
    call = _mock_tool_call(name="terminal", arguments=json.dumps(payload), call_id="c1")
    messages = []

    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)

    assert events == []
    assert len(messages) == 1
    result = json.loads(messages[0]["content"])
    assert result["error_type"] == "incomplete_historical_tool_arguments"
    assert messages[0]["effect_disposition"] == "none"


@pytest.mark.parametrize("concurrent", [False, True])
def test_interrupt_rejects_deferred_incomplete_arguments(agent, monkeypatch, concurrent):
    from tools import tool_search
    from tools.registry import registry

    events = []
    registry.register(
        name="mcp__integrity_probe__run",
        toolset="mcp-integrity-probe",
        schema={
            "name": "mcp__integrity_probe__run",
            "description": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kwargs: events.append("handler") or "handled",
    )
    agent._interrupt_requested = True
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *args, **kwargs: events.append("hook") or [],
    )
    outer = {
        "name": "mcp__integrity_probe__run",
        "arguments": json.dumps(
            {RESERVED: {"version": 1, "replayable": False}}
        ),
    }
    call = _mock_tool_call(
        name=tool_search.TOOL_CALL_NAME,
        arguments=json.dumps(outer),
        call_id="deferred-interrupted",
    )
    messages = []
    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)
    assert events == []
    assert json.loads(messages[0]["content"])["error_type"] == "incomplete_historical_tool_arguments"


@pytest.mark.parametrize("concurrent", [False, True])
def test_preexisting_interrupt_cancels_malformed_json_consistently(
    agent, monkeypatch, concurrent
):
    terminal_events = []
    agent._interrupt_requested = True
    monkeypatch.setattr(
        "agent.tool_executor._emit_terminal_post_tool_call",
        lambda *_args, **kwargs: terminal_events.append(kwargs),
    )
    call = _mock_tool_call(name="terminal", arguments="{broken", call_id="c1")
    messages = []

    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)

    assert len(messages) == 1
    assert "cancelled" in messages[0]["content"]
    assert "invalid_tool_arguments" not in messages[0]["content"]
    assert messages[0]["effect_disposition"] == "none"
    assert len(terminal_events) == 1
    assert terminal_events[0]["status"] == "cancelled"
    assert terminal_events[0]["error_type"] == "user_interrupt"


def test_concurrent_preexisting_interrupt_flushes_integrity_rejection(
    agent, monkeypatch
):
    agent._interrupt_requested = True
    payload = {RESERVED: {"version": 1, "replayable": False}}
    call = _mock_tool_call(name="terminal", arguments=json.dumps(payload), call_id="c1")
    messages = []
    flushes = []
    monkeypatch.setattr(
        "agent.tool_executor._flush_session_db_after_tool_progress",
        lambda _agent, current, **_kwargs: flushes.append(list(current)) or True,
    )

    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, True, call, messages)

    assert len(flushes) == 1
    assert flushes[0] == messages
    assert json.loads(messages[0]["content"])["error_type"] == "incomplete_historical_tool_arguments"


def test_concurrent_preexisting_interrupt_flushes_mixed_results_in_order(
    agent, monkeypatch
):
    agent._interrupt_requested = True
    valid = _mock_tool_call(name="terminal", arguments='{"command":"true"}', call_id="v")
    payload = {RESERVED: {"version": 1, "replayable": False}}
    invalid = _mock_tool_call(name="terminal", arguments=json.dumps(payload), call_id="i")
    message = _mock_assistant_msg(content="", tool_calls=[valid, invalid])
    messages = []
    flushes = []
    monkeypatch.setattr(
        "agent.tool_executor._flush_session_db_after_tool_progress",
        lambda _agent, current, **_kwargs: flushes.append(
            [entry["tool_call_id"] for entry in current]
        )
        or True,
    )

    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        agent._execute_tool_calls_concurrent(message, messages, "task-1")

    assert flushes == [["v"], ["v", "i"]]
    assert [entry["tool_call_id"] for entry in messages] == ["v", "i"]
    assert json.loads(messages[1]["content"])["error_type"] == "incomplete_historical_tool_arguments"


@pytest.mark.parametrize("concurrent", [False, True])
def test_clean_arguments_skip_schema_integrity_preview(
    agent, monkeypatch, concurrent
):
    call = _mock_tool_call(
        name="web_search",
        arguments=json.dumps({"query": "Hermes"}),
        call_id="clean",
    )
    messages = []
    monkeypatch.setattr(
        "agent.tool_executor._schema_decoded_integrity_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("schema preview should be skipped")
        ),
    )
    with patch("run_agent.handle_function_call", return_value="ok"):
        _run(agent, concurrent, call, messages)
    assert len(messages) == 1
    assert messages[0]["content"] == "ok"
