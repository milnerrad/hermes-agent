"""Actual same-owner in-flight verdict handoff: cooldown and cancellation regressions."""
import asyncio
from types import SimpleNamespace
import pytest
from agent.lsp import deferred, edit_check
from tools.file_operations_lint import _queue_late_verdict

PATH = '/disposable/inflight-owner.ts'
ERROR = [{'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 1}},
          'severity': 1, 'code': 2322, 'message': 'retained predecessor sentinel'}]


class Service:
    def __init__(self):
        self._edit_checks = edit_check.EditChecks()
        self.cooldown = False
    def _edit_diagnostics_epoch(self, path):
        return ('workspace', 0)
    def _skip_degraded_diagnostics(self, path):
        return self.cooldown
    def _wait_budget(self, path):
        return .1
    def _finish_edit_diagnostics(self, epoch, ok):
        pass
    async def _mark_broken_for_file_async(self, path, exc):
        self.cooldown = True


def queued(owner, task):
    task.add_done_callback(lambda f: _queue_late_verdict(owner, PATH, f, (), 'oldest clean baseline'))
    return task


def notes(owner):
    return deferred.settle_for_agent(SimpleNamespace(session_id=owner), timeout=0)


@pytest.mark.asyncio
async def test_cooldown_replacement_retains_real_predecessor_and_baseline(monkeypatch):
    svc = Service(); owner = 'cooldown-handoff-owner'
    entered, release = asyncio.Event(), asyncio.Event()
    async def check(svc, path, pre, post, budget):
        if post == 'A':
            entered.set(); await release.wait(); return ERROR
        if path.endswith('other.ts'):
            svc.cooldown = True; return None
        pytest.fail('replacement must honor newly opened cooldown')
    monkeypatch.setattr(edit_check, '_check', check)
    a = queued(owner, asyncio.create_task(edit_check._serialized(svc, PATH, 'oldest clean baseline', 'A', owner, ())))
    await entered.wait()
    b = queued(owner, asyncio.create_task(edit_check._serialized(svc, PATH, 'intermediate error', 'B', owner, ())))
    await asyncio.sleep(0)
    await edit_check._serialized(svc, '/disposable/other.ts', None, 'other', 'other-owner', ())
    release.set(); await a; result = await b; await asyncio.sleep(0)
    assert result.incomplete and result.diagnostics == ERROR
    assert result.baseline_text == 'oldest clean baseline'
    delivered = str(notes(owner))
    assert delivered.count('retained predecessor sentinel') == 1
    assert 'NOT verified clean' in delivered
    assert notes('unrelated-owner') == []


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['waiting', 'own-check', 'successor-waiting'])
async def test_cancellation_handoff_retains_inflight_error_exactly_once(monkeypatch, phase):
    svc = Service(); owner = 'cancel-handoff-' + phase
    entered, release, b_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def check(svc, path, pre, post, budget):
        if post == 'A':
            entered.set(); await release.wait(); return ERROR
        if post == 'B':
            b_entered.set(); await asyncio.Event().wait()
        if post == 'C':
            return None
        return []
    monkeypatch.setattr(edit_check, '_check', check)
    a = queued(owner, asyncio.create_task(edit_check._serialized(svc, PATH, 'oldest clean baseline', 'A', owner, ())))
    await entered.wait()
    b = queued(owner, asyncio.create_task(edit_check._serialized(svc, PATH, 'intermediate error', 'B', owner, ())))
    await asyncio.sleep(0)
    if phase == 'own-check':
        release.set(); await b_entered.wait()
    c = None
    if phase == 'successor-waiting':
        c = queued(owner, asyncio.create_task(edit_check._serialized(svc, PATH, 'later unknown', 'C', owner, ())))
        await asyncio.sleep(0)
    b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b
    release.set(); await a
    if c is not None:
        await c
    await asyncio.sleep(0)
    delivered = str(notes(owner))
    assert delivered.count('retained predecessor sentinel') == 1
    assert 'NOT verified clean' in delivered
    assert notes('unrelated-owner') == []
    # A later actual clean correction consumes carried evidence; never resurrect it.
    deferred.post(owner, PATH, 'retained predecessor sentinel; NOT verified clean', 'oldest clean baseline')
    carry = (deferred.take(owner, PATH),)
    clean = await edit_check._serialized(svc, PATH, 'error before correction', 'clean', owner, carry)
    assert clean.diagnostics == [] and not clean.incomplete
    assert notes(owner) == []
