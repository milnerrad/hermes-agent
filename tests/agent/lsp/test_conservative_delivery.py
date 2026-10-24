"""Conservative production delivery boundary regressions."""
import inspect
from unittest.mock import Mock
from tools.file_operations import ShellFileOperations
from tools.file_operations import ExecuteResult


def test_safe_shell_writer_even_on_local_backend():
    ops = object.__new__(ShellFileOperations)
    ops._native_read_enabled = lambda: True
    ops._exec = Mock(return_value=ExecuteResult(stdout='', exit_code=0))
    ops._atomic_write('/nonexistent/conservative.ts', 'new bytes')
    assert ops._exec.called, 'write must use original safe shell transport'
    assert 'mktemp' in ops._exec.call_args.args[0]


def test_safe_probe_and_hash_stay_shell_based():
    assert '_native_io' not in inspect.getsource(ShellFileOperations._probe_write_target)
    assert '_native_io' not in inspect.getsource(ShellFileOperations._verify_written_hash)

from tests.agent.lsp.test_write_never_waits_on_lsp import mock_workspace, _patch
from agent.lsp import deferred, get_service
from types import SimpleNamespace


def test_async_write_respects_active_cooldown(mock_workspace):
    repo, gate = mock_workspace('by_content')
    import agent.lsp
    svc = agent.lsp.get_service()
    path = repo / 'cooldown.py'
    path.write_text('x = 1\n')
    svc._degrade_diagnostics(str(path))
    value, _ = _patch(path, 'x = 1', 'x = bad', session='cooldown-new')
    notes = deferred.settle_for_agent(SimpleNamespace(session_id='cooldown-new'), timeout=.1)
    assert not deferred._PENDING.get('cooldown-new'), 'cooldown must not enqueue server work'
    assert 'unavailable' in (str(notes) + str(value)).lower()
    assert not svc._clients


def test_expired_pending_is_unknown_exactly_once():
    owner = 'expired-conservative'
    token = deferred.track_pending(owner, '/expired.ts', timeout=0)
    try:
        notes = deferred.settle_for_agent(SimpleNamespace(session_id=owner))
        assert 'NOT verified clean' in str(notes)
        assert deferred.settle_for_agent(SimpleNamespace(session_id=owner)) == []
    finally:
        deferred.settled(owner, token)


def test_cold_final_budget_is_fifteen_not_five(monkeypatch):
    owner = 'cold-budget-conservative'
    token = deferred.track_pending(owner, '/cold.ts', timeout=15)
    observed = []
    monkeypatch.setattr(deferred._SETTLED, 'wait_for', lambda predicate, timeout: observed.append(timeout))
    try:
        deferred.settle_for_agent(SimpleNamespace(session_id=owner))
        assert 14 < observed[0] <= 15
    finally:
        deferred.settled(owner, token)


def test_cap_still_discloses_without_model_continuation(monkeypatch):
    from agent.turn_stop_gates import apply_stop_gates
    agent = SimpleNamespace(session_id='cap-conservative', _lsp_turn_end_notes=3)
    downstream = []
    monkeypatch.setattr('agent.turn_stop_gates._verify_on_stop_nudge', lambda _: downstream.append('verified'))
    monkeypatch.setattr('agent.turn_stop_gates._pre_verify_nudge', lambda *a: None)
    monkeypatch.setattr('agent.turn_stop_gates._kanban_stop_nudge', lambda *a: None)
    deferred.post(agent.session_id, '/unresolved.ts', 'diagnostics unavailable; NOT verified clean', None)
    final = {'role': 'assistant', 'content': 'done'}
    verdict = apply_stop_gates(agent, final, final_response='done', messages=[], conversation_history=[],
                              pending_verification_response=None, pending_verification_response_previewed=False)
    assert not verdict.continue_turn
    assert 'NOT verified clean' in verdict.final_response
    assert final['content'] == verdict.final_response
    assert downstream == ['verified'], 'LSP cap cannot bypass independent stop gates'


def test_cancelled_check_is_unknown_not_clean():
    from concurrent.futures import Future
    from tools.file_operations_lint import _post_deferred
    owner = 'cancel-conservative'
    future = Future(); future.cancel()
    token = deferred.track_pending(owner)
    _post_deferred(owner, '/cancelled.ts', future, (), token)
    assert 'NOT verified clean' in str(deferred.settle_for_agent(SimpleNamespace(session_id=owner)))


def test_async_cooldown_probe_recovers_without_suppressing_healthy_checks(mock_workspace, monkeypatch):
    from unittest.mock import AsyncMock
    import agent.lsp
    import time
    repo, _ = mock_workspace('by_content')
    svc = agent.lsp.get_service()
    path = str(repo / 'recovery.py')
    check = AsyncMock(return_value=[])
    monkeypatch.setattr('agent.lsp.edit_check._check', check)
    assert svc.start_edit_check(path, pre_text=None, post_text='x=1', owner='probe').result(2).diagnostics == []
    assert check.await_count == 1
    svc._degrade_diagnostics(path)
    assert svc.start_edit_check(path, pre_text=None, post_text='x=2', owner='probe').result(2).diagnostics is None
    assert check.await_count == 1
    key = next(iter(svc._diagnostics_degraded_until))
    svc._diagnostics_degraded_until[key] = time.monotonic() - 1
    assert svc.start_edit_check(path, pre_text=None, post_text='x=3', owner='probe').result(2).diagnostics == []
    assert check.await_count == 2
    assert not svc._diagnostics_degraded_until and not svc._diagnostics_probe_inflight


def test_older_healthy_check_cannot_close_new_timeout_circuit(mock_workspace):
    import agent.lsp
    repo, _ = mock_workspace('by_content')
    svc = agent.lsp.get_service()
    path = str(repo / 'race.py')
    epoch = svc._edit_diagnostics_epoch(path)
    svc._degrade_diagnostics(path)
    expected = dict(svc._diagnostics_degraded_until)
    svc._finish_edit_diagnostics(epoch, True)
    assert svc._diagnostics_degraded_until == expected
