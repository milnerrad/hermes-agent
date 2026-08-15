"""Fail-closed execution tests for compressed historical tool arguments."""

from __future__ import annotations

import json

import pytest

from agent.tool_executor import _parse_tool_arguments


RESERVED = "__hermes_incomplete_tool_arguments__"


@pytest.mark.parametrize(
    "shape",
    [
        {"command": "python -c 'print(1)'"},
        {"path": "/tmp/a", "content": "payload"},
        {"mode": "replace", "path": "/tmp/a", "old_string": "a", "new_string": "b"},
    ],
)
def test_incomplete_provenance_blocks_operation_shapes(shape):
    shape[RESERVED] = {
        "version": 1,
        "reason": "context_compression",
        "replayable": False,
    }
    arguments, error = _parse_tool_arguments(json.dumps(shape))

    assert arguments == {}
    parsed_error = json.loads(error or "{}")
    assert parsed_error["error_type"] == "incomplete_historical_tool_arguments"
    assert "not executed" in parsed_error["message"]


def test_nested_or_malformed_reserved_provenance_fails_closed():
    for payload in (
        {"outer": [{RESERVED: {"version": 1}}]},
        {RESERVED: "malformed"},
        {RESERVED: None},
    ):
        arguments, error = _parse_tool_arguments(json.dumps(payload))
        assert arguments == {}
        assert json.loads(error or "{}")["error_type"] == "incomplete_historical_tool_arguments"


def test_literal_truncation_marker_is_not_treated_as_provenance():
    payload = {
        "path": "/tmp/a",
        "content": "Documentation containing ...[truncated] literally.",
    }
    arguments, error = _parse_tool_arguments(json.dumps(payload))

    assert error is None
    assert arguments == payload
