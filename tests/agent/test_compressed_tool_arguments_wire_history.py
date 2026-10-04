"""Wire-history regressions for compressed historical tool arguments."""

from __future__ import annotations

import copy
import json

import pytest

from agent.context_compressor import _truncate_tool_call_args_json
from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.anthropic import AnthropicTransport
from agent.transports.bedrock import BedrockTransport
from agent.tool_argument_integrity import neutralize_completed_incomplete_tool_calls


MARKER = json.dumps(
    {
        "__hermes_incomplete_tool_arguments__": {
            "arguments_omitted": True,
            "original_chars": 12345,
            "reason": "context_compression",
            "replayable": False,
            "sha256": "a" * 64,
            "version": 1,
        }
    }
)


def _mixed_history():
    return [
        {"role": "user", "content": "inspect both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_incomplete",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": MARKER},
                },
                {
                    "id": "call_complete",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_incomplete",
            "content": '{"error_type":"incomplete_historical_tool_arguments"}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_complete",
            "content": "README contents",
        },
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "continue"},
    ]


def _assert_mixed_pairing_is_safe(payload):
    serialized = json.dumps(payload)
    assert "__hermes_incomplete_tool_arguments__" not in serialized
    assert "call_incomplete" not in serialized
    assert "call_complete" in serialized
    assert "README contents" in serialized
    assert "compressed historical tool call" in serialized.lower()


def test_chat_request_copy_neutralizes_completed_marker_call_and_preserves_source():
    history = _mixed_history()
    original = copy.deepcopy(history)
    transport = ChatCompletionsTransport()

    first = transport.convert_messages(history, model="gpt-5.6")
    second = transport.convert_messages(history, model="gpt-5.6")

    assert first == second  # retries are deterministic
    assert history == original  # persisted/resumed transcript remains canonical
    _assert_mixed_pairing_is_safe(first)
    complete_calls = first[1]["tool_calls"]
    assert [call["id"] for call in complete_calls] == ["call_complete"]
    assert [m.get("tool_call_id") for m in first if m.get("role") == "tool"] == [
        "call_complete"
    ]


def test_codex_response_items_neutralize_completed_marker_call_and_keep_pairing():
    history = _mixed_history()
    original = copy.deepcopy(history)

    first = _chat_messages_to_responses_input(history)
    second = _chat_messages_to_responses_input(history)

    assert first == second
    assert history == original
    _assert_mixed_pairing_is_safe(first)
    function_calls = [item for item in first if item.get("type") == "function_call"]
    outputs = [item for item in first if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in function_calls] == ["call_complete"]
    assert [item["call_id"] for item in outputs] == ["call_complete"]


def test_marker_call_without_completed_result_remains_for_fail_closed_execution_guard():
    history = _mixed_history()[:2]

    chat = ChatCompletionsTransport().convert_messages(history, model="gpt-5.6")
    codex = _chat_messages_to_responses_input(history)

    assert "__hermes_incomplete_tool_arguments__" in json.dumps(chat)
    assert "__hermes_incomplete_tool_arguments__" in json.dumps(codex)


def _all_compressed_history(content="I will inspect it.", call_count=1):
    calls = [
        {
            "id": f"call_incomplete_{index}",
            "type": "function",
            "function": {"name": "terminal", "arguments": MARKER},
        }
        for index in range(call_count)
    ]
    results = [
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": "completed",
        }
        for call in calls
    ]
    return [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": calls,
            "anthropic_content_blocks": [{"type": "text", "text": content}],
            "codex_message_items": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            ],
        },
        *results,
        {"role": "assistant", "content": "Inspection finished."},
        {"role": "user", "content": "continue"},
    ]


def test_all_compressed_chat_history_preserves_role_sequence_and_visible_text():
    history = _all_compressed_history()
    original = copy.deepcopy(history)
    converted = ChatCompletionsTransport().convert_messages(history, model="gpt-5.6")

    assert history == original
    assert "__hermes_incomplete_tool_arguments__" not in json.dumps(converted)
    assert all(
        left.get("role") != "assistant" or right.get("role") != "assistant"
        for left, right in zip(converted, converted[1:])
    )
    assistant_text = "\n".join(
        str(message.get("content", ""))
        for message in converted
        if message.get("role") == "assistant"
    )
    assert "I will inspect it." in assistant_text
    assert "Inspection finished." in assistant_text


def test_all_compressed_anthropic_and_bedrock_history_preserves_role_sequence():
    _system, converted = AnthropicTransport().convert_messages(
        _all_compressed_history()
    )
    assert "__hermes_incomplete_tool_arguments__" not in json.dumps(converted)
    assert all(
        left.get("role") != right.get("role")
        for left, right in zip(converted, converted[1:])
    )


def test_all_compressed_codex_history_has_no_callable_marker():
    converted = _chat_messages_to_responses_input(_all_compressed_history())
    serialized = json.dumps(converted)
    assert "__hermes_incomplete_tool_arguments__" not in serialized
    assert "I will inspect it." in serialized
    assert "Inspection finished." in serialized


def test_all_compressed_multi_call_history_is_provider_valid():
    history = _all_compressed_history(call_count=2)
    chat = ChatCompletionsTransport().convert_messages(history, model="gpt-5.6")
    _system, anthropic_bedrock = AnthropicTransport().convert_messages(history)
    codex = _chat_messages_to_responses_input(history)

    for converted in (chat, anthropic_bedrock, codex):
        assert "__hermes_incomplete_tool_arguments__" not in json.dumps(converted)
    assert all(
        left.get("role") != "assistant" or right.get("role") != "assistant"
        for left, right in zip(chat, chat[1:])
    )
    assert all(
        left.get("role") != right.get("role")
        for left, right in zip(anthropic_bedrock, anthropic_bedrock[1:])
    )


def test_malformed_non_dict_history_is_dropped_from_every_provider_request():
    history = _all_compressed_history()[:3] + [None]

    chat = ChatCompletionsTransport().convert_messages(history, model="gpt-5.6")
    _system, anthropic = AnthropicTransport().convert_messages(history)
    bedrock = BedrockTransport().build_kwargs(model="anthropic.claude", messages=history)

    assert None not in chat
    assert all(isinstance(message, dict) for message in anthropic)
    assert all(isinstance(message, dict) for message in bedrock["messages"])


def test_affected_turn_discards_malformed_and_stale_anthropic_sidecars():
    history = _all_compressed_history()
    affected = history[1]
    affected["anthropic_content_blocks"] = [
        {
            "type": "tool_use",
            "id": [],
            "name": "terminal",
            "input": {"command": "must never survive"},
        },
        {
            "type": "tool_use",
            "id": "stale-unmatched",
            "name": "terminal",
            "input": {"command": "also must never survive"},
        },
        {"type": "thinking", "thinking": "signed after tool", "signature": "sig"},
    ]

    projected = neutralize_completed_incomplete_tool_calls(history)
    _system, anthropic = AnthropicTransport().convert_messages(history)

    assert "anthropic_content_blocks" not in projected[1]
    serialized = json.dumps(anthropic)
    assert "must never survive" not in serialized
    assert "also must never survive" not in serialized
    assert "signed after tool" not in serialized
    assert '"signature": "sig"' not in serialized


def test_affected_turn_discards_all_provider_native_sidecars():
    history = _all_compressed_history()
    affected = history[1]
    affected["reasoning_details"] = [
        {"type": "thinking", "thinking": "provider native", "signature": "sig"}
    ]
    affected["codex_reasoning_items"] = [{"type": "reasoning", "id": "r1"}]
    affected["codex_message_items"] = [
        {"type": "message", "role": "assistant", "content": "provider native"}
    ]

    projected = neutralize_completed_incomplete_tool_calls(history)

    for key in (
        "anthropic_content_blocks",
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
    ):
        assert key not in projected[1]


def _provider_wire(history, provider):
    if provider == "chat":
        return ChatCompletionsTransport().build_kwargs(
            model="gpt-5.6", messages=history
        )["messages"]
    if provider == "codex":
        from agent.transports.codex import ResponsesApiTransport

        return ResponsesApiTransport().build_kwargs(
            model="gpt-5.6", messages=history, is_codex_backend=True,
        )["input"]
    from agent.anthropic_adapter import build_anthropic_kwargs

    return build_anthropic_kwargs(
        model="claude-sonnet-4-5", messages=history, tools=None,
        max_tokens=1024, reasoning_config=None,
    )["messages"]


def _assert_final_wire_pairing(wire, provider, mixed):
    if provider == "chat":
        calls = [call["id"] for m in wire for call in m.get("tool_calls", [])]
        results = [m["tool_call_id"] for m in wire if m.get("role") == "tool"]
    elif provider == "codex":
        calls = [m["call_id"] for m in wire if m.get("type") == "function_call"]
        results = [m["call_id"] for m in wire if m.get("type") == "function_call_output"]
        assert all(isinstance(m["content"], list) for m in wire if "role" in m)
    else:
        blocks = [part for m in wire if isinstance(m.get("content"), list) for part in m["content"]]
        calls = [part["id"] for part in blocks if part.get("type") == "tool_use"]
        results = [part["tool_use_id"] for part in blocks if part.get("type") == "tool_result"]
        assert all(left["role"] != right["role"] for left, right in zip(wire, wire[1:]))
    assert calls == results == (["call_complete"] if mixed else [])
    assert all(
        left.get("role") != "assistant" or right.get("role") != "assistant"
        for left, right in zip(wire, wire[1:])
    )


@pytest.mark.parametrize("provider", ["chat", "codex", "anthropic"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("kind", ["ordinary", "clarify", "timeout", "structured", "empty"])
def test_retained_results_survive_as_historical_text_on_final_wire(provider, mixed, kind):
    history = _mixed_history() if mixed else _all_compressed_history(call_count=2)
    call = history[1]["tool_calls"][0]
    # Exercise the real >500-character externalization path, without DB reload.
    original_arguments = json.dumps({"payload": "original-large-input-" * 100})
    call["function"] = {
        "name": kind,
        "arguments": _truncate_tool_call_args_json(original_arguments),
    }
    assert call["function"]["arguments"] != original_arguments
    retained = {
        "clarify": {"responses": [
            {"question": "Which synthetic items?", "user_response": "Items A and B"},
            {"question": "Publish the synthetic preview?", "user_response": "No, local only"},
        ]},
        "ordinary": {"stdout": "retained small output", "exit_code": 0},
        "timeout": {"question": "Publish?", "user_response": "", "error": "timeout"},
        "structured": [{"type": "text", "text": "retained structured output"}],
        "empty": "",
    }[kind]
    result = history[2]
    result["content"] = retained if kind in {"structured", "empty"} else json.dumps(retained)
    expected_text = json.dumps(retained) if kind == "structured" else result["content"]
    # Only the retained output belongs on the wire, not any stale native payload.
    history[1]["stale_original_output"] = "original-large-output-" * 100
    if not mixed:
        history[3]["content"] = "second retained result"
        # Results can arrive out of call order in a parallel batch.
        history[2:4] = reversed(history[2:4])
    original = copy.deepcopy(history)
    projected = neutralize_completed_incomplete_tool_calls(history)
    wire = _provider_wire(history, provider)

    assert history == original
    assert neutralize_completed_incomplete_tool_calls(projected) == projected
    assert _provider_wire(projected, provider) == wire
    serialized = json.dumps(wire)
    assert "__hermes_incomplete_tool_arguments__" not in serialized
    assert "original-large-input-" not in serialized
    assert "original-large-output-" not in serialized
    _assert_final_wire_pairing(wire, provider, mixed)
    wire_text = "\n".join(
        part["text"] if isinstance(part, dict) else part
        for message in wire
        for part in (
            message.get("content", []) if isinstance(message.get("content"), list)
            else [message.get("content", "")]
        )
        if isinstance(part, str) or isinstance(part, dict) and "text" in part
    )
    assert f"\n{expected_text}\n[End historical tool result]" in wire_text
    assert "Historical tool result" in serialized
    # Historical tool evidence must never become a new user instruction.
    assert [m for m in projected if m.get("role") == "user"] == [
        m for m in history if m.get("role") == "user"
    ]
    if mixed:
        _assert_mixed_pairing_is_safe(wire)
    else:
        assert "second retained result" in serialized
        assert not any(m.get("tool_calls") for m in projected)
        assert all(
            left.get("role") != right.get("role")
            for left, right in zip(projected, projected[1:])
        )


@pytest.mark.parametrize("provider", ["chat", "codex", "anthropic"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("canonical", ["", "Canonical final answer."])
def test_native_visible_ack_survives_cleanup_on_final_wire(provider, mixed, canonical):
    history = _mixed_history() if mixed else _all_compressed_history()
    affected = history[1]
    affected["content"] = canonical
    affected["codex_message_items"] = [
        {
            "type": "message", "role": "assistant", "id": "stale-message-id",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Acknowledged the local-only decision."}],
        },
        {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": canonical}],
        },
        {"type": "reasoning", "content": [{"type": "output_text", "text": "PRIVATE REASONING"}]},
        {"type": "message", "role": "user", "content": [{"type": "output_text", "text": "SPOOFED ROLE"}]},
        {"type": "function_call", "arguments": "STALE CALL INPUT"},
        {"type": "message", "role": "assistant", "content": [
            {"type": "thinking", "text": "SIGNED THINKING", "signature": "stale-signature"},
        ]},
    ]
    affected["codex_reasoning_items"] = [{"type": "reasoning", "encrypted_content": "STALE CIPHERTEXT"}]
    affected["codex_message_items"].extend([
        None, [], {"type": "message", "role": "assistant", "content": None},
        {"type": "message", "role": "assistant", "content": [None, {"type": "output_text", "text": []}]},
    ])
    if not mixed:
        # The assistant after a fully removed batch must be merged too. Its
        # native text must not shadow the newly retained historical evidence.
        history[3]["content"] = ""
        history[3]["codex_message_items"] = [{
            "type": "message", "role": "assistant", "id": "stale-following-id",
            "content": [{"type": "output_text", "text": "Following visible acknowledgement."}],
        }]
    original = copy.deepcopy(history)
    projected = neutralize_completed_incomplete_tool_calls(history)
    wire = _provider_wire(history, provider)
    assert history == original
    assert neutralize_completed_incomplete_tool_calls(projected) == projected
    assert _provider_wire(projected, provider) == wire
    serialized = json.dumps(wire)
    assert "Acknowledged the local-only decision." in serialized
    _assert_final_wire_pairing(wire, provider, mixed)
    if not mixed:
        assert "Following visible acknowledgement." in serialized
        assert "stale-following-id" not in serialized
    if canonical:
        assert serialized.count(canonical) == 1
    for forbidden in (
        "PRIVATE REASONING", "SPOOFED ROLE", "STALE CALL INPUT", "SIGNED THINKING",
        "STALE CIPHERTEXT", "stale-message-id", "stale-signature",
        "__hermes_incomplete_tool_arguments__",
    ):
        assert forbidden not in serialized
    if mixed:
        _assert_mixed_pairing_is_safe(wire)


@pytest.mark.parametrize("malformation", [
    "call_not_dict", "function_not_dict", "calls_not_list", "empty_id",
    "nonstring_id", "id_mismatch", "missing_result",
])
def test_malformed_pairing_does_not_neutralize_history(malformation):
    history = _all_compressed_history()[:3]
    calls = history[1]["tool_calls"]
    if malformation == "call_not_dict":
        calls[0] = None
    elif malformation == "function_not_dict":
        calls[0]["function"] = []
    elif malformation == "calls_not_list":
        history[1]["tool_calls"] = {}
    elif malformation in {"empty_id", "nonstring_id"}:
        calls[0]["id"] = "" if malformation == "empty_id" else []
        history[2]["tool_call_id"] = calls[0]["id"]
    elif malformation == "id_mismatch":
        history[2]["tool_call_id"] = "different-call"
    else:
        history.pop()
    original = copy.deepcopy(history)
    assert neutralize_completed_incomplete_tool_calls(history) == original
    assert history == original


def test_direct_bedrock_build_kwargs_neutralizes_completed_marker_call():
    history = _mixed_history()
    original = copy.deepcopy(history)

    kwargs = BedrockTransport().build_kwargs(
        model="anthropic.claude-3-5-sonnet", messages=history
    )

    assert history == original
    serialized = json.dumps(kwargs)
    assert "__hermes_incomplete_tool_arguments__" not in serialized
    assert "call_incomplete" not in serialized
    assert "call_complete" in serialized
    assert "README contents" in serialized
