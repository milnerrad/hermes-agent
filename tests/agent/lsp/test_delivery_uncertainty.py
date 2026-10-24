"""Unknown verdicts retain provenance and never silently certify an edit."""
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from agent.lsp import deferred
from agent.lsp.edit_check import CheckResult
from tools.file_operations_lint import LintMixin, _post_deferred


def test_unknown_replacement_keeps_immutable_baseline():
    owner, path = "unknown-baseline", "/disposable/unknown.py"
    future = Future()
    future.set_result(CheckResult(None, "pre-existing error baseline"))
    token = deferred.track_pending(owner)
    _post_deferred(owner, path, future, (), token)
    note = deferred.take(owner, path)
    assert note is not None and "NOT verified clean" in note.text
    assert note.baseline_text == "pre-existing error baseline"


def test_cancelled_replacement_keeps_immutable_baseline():
    owner, path = "cancel-baseline", "/disposable/cancel.py"
    future = Future()
    future.cancel()
    token = deferred.track_pending(owner)
    _post_deferred(owner, path, future, (), token, "original baseline")
    note = deferred.take(owner, path)
    assert note is not None and "NOT verified clean" in note.text
    assert note.baseline_text == "original baseline"


def test_ownerless_timeout_is_disclosed_inline(monkeypatch):
    ops = object.__new__(LintMixin)
    svc = Mock()
    svc.start_edit_check.return_value = Future()  # no callback owner can deliver it later
    monkeypatch.setattr(ops, "_lsp_service", lambda: svc)
    monkeypatch.setattr(ops, "_lsp_will_handle", lambda path: True)
    monkeypatch.setattr("agent.lsp.edit_check.inline_wait", lambda _: 0)
    with deferred.bind_owner(None):
        text = ops._maybe_lsp_diagnostics("/disposable/no-owner.ts", pre_content="before", post_content="after")
    assert "NOT verified clean" in text


def test_pending_uncertainty_cannot_cross_session_boundary():
    token = deferred.track_pending("pending-owner", "/disposable/owned.ts", timeout=0)
    try:
        assert deferred.settle_for_agent(SimpleNamespace(session_id="other-owner"), timeout=0) == []
        notes = deferred.settle_for_agent(SimpleNamespace(session_id="pending-owner"), timeout=0)
        assert len(notes) == 1 and "NOT verified clean" in notes[0][1]
        deferred.post("pending-owner", "/disposable/owned.ts", "late actual error", "before")
        assert deferred.settle_for_agent(SimpleNamespace(session_id="other-owner"), timeout=0) == []
        assert deferred.settle_for_agent(SimpleNamespace(session_id="pending-owner"), timeout=0)[0][1] == "late actual error"
    finally:
        deferred.settled("pending-owner", token)


def test_cap_error_only_discloses_unknown_and_runs_independent_gate(monkeypatch):
    from agent.turn_stop_gates import apply_stop_gates
    import agent.turn_stop_gates as gates
    agent = SimpleNamespace(session_id="cap-errors", _lsp_turn_end_notes=3,
                            _interim_content_was_streamed=lambda _: False,
                            _emit_interim_assistant_message=lambda _: None,
                            _flush_messages_to_session_db=lambda *args: None)
    monkeypatch.setattr(gates, "_verify_on_stop_nudge", lambda _: "independent verification still required")
    deferred.post(agent.session_id, "/disposable/error.ts", "TS2322 error", "before")
    messages = []
    final = {"role": "assistant", "content": "done"}
    verdict = apply_stop_gates(agent, final, final_response="done", messages=messages, conversation_history=[],
                              pending_verification_response=None, pending_verification_response_previewed=False)
    assert verdict.continue_turn and verdict.final_response is None
    assert "NOT verified clean" in verdict.pending_verification_response
    assert messages[-1]["content"] == "independent verification still required"
