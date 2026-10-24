"""Deferred delivery safety: real non-TS mock protocol; deterministic model seam."""
from concurrent.futures import Future
from types import SimpleNamespace
from agent.lsp import deferred
from tests.agent.lsp.test_write_never_waits_on_lsp import mock_workspace, _patch


def test_corrected_non_ts_edit_does_not_report_old_error(mock_workspace):
    repo, gate = mock_workspace('by_content')
    target = repo / 'corrected.py'
    target.write_text('x = 1\n')
    _patch(target, 'x = 1', 'x = bad', session='corrected-owner')
    _patch(target, 'x = bad', 'x = 1', session='corrected-owner')
    gate.touch()
    agent = SimpleNamespace(session_id='corrected-owner', _current_task_id=None)
    assert deferred.settle_for_agent(agent) == []
    assert not deferred._PENDING.get('corrected-owner')
    assert target.read_text() == 'x = 1\n'


def test_cancelled_replacement_restores_undelivered_note():
    from tools.file_operations_lint import _post_deferred
    owner, path = 'cancel-owner', '/disposable/never-written.ts'
    deferred.post(owner, path, '2322 original real-error note', 'clean baseline')
    previous = deferred.take(owner, path)
    token = deferred.track_pending(owner)
    future = Future()
    future.cancel()
    _post_deferred(owner, path, future, (previous,), token)
    notes = deferred.settle_for_agent(SimpleNamespace(session_id=owner))
    assert notes[0][0] == path and '2322 original real-error note' in notes[0][1]
    assert 'NOT verified clean' in notes[0][1]
    assert not deferred._PENDING.get(owner)
