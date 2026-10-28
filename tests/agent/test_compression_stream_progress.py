"""Sanitized per-attempt Codex summary progress, including abandoned workers."""
import json
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import compression_facade as facade
from agent import conversation_compression as cc
from hermes_cli import config
from tests.agent.test_native_backup_timeout import BackupAgent, native_gate


def test_codex_events_separate_liveness_reasoning_and_semantic_summary():
    from agent.auxiliary_stream_progress import AuxStreamProgress
    progress = AuxStreamProgress()
    guard = aux._CodexStreamGuard(SimpleNamespace(base_url="offline"), 300)
    with aux.aux_stream_observation_hook(progress.observe):
        for event in [
            {"type": "response.in_progress", "secret": "never serialize this"},
            {"type": "response.reasoning_summary_text.delta", "delta": "秘密"},
            {"type": "response.output_text.delta", "delta": " \n\t"},
            {"type": "response.output_text.delta", "delta": "useful summary"},
            {"type": "response.completed", "response": {"encrypted_content": "opaque"}},
        ]:
            guard.on_event(event)
    stats = progress.snapshot()
    assert stats["stream_events"] == 5
    assert stats["stream_reasoning_events"] == 1 and stats["stream_reasoning_bytes"] == 6
    assert stats["stream_whitespace_events"] == 1 and stats["stream_whitespace_bytes"] == 3
    assert stats["stream_summary_events"] == 1 and stats["stream_summary_bytes"] == 14
    assert stats["stream_completed"] is True
    assert stats["stream_first_event_ms"] <= stats["stream_first_reasoning_ms"] <= stats["stream_first_summary_ms"]
    assert all(isinstance(v, (int, bool)) or v is None for v in stats.values())
    assert not any(s in json.dumps(stats) for s in ["秘密", "useful summary", "opaque", "secret"])


@pytest.mark.parametrize("cancel", [False, True])
def test_progress_record_emitted_even_if_host_does_not_wait_for_worker(native_gate, monkeypatch, cancel):
    from agent.auxiliary_stream_progress import AuxStreamProgress
    monkeypatch.setattr(config, "load_config", lambda: {"compression": {}})
    records = []
    monkeypatch.setattr("agent.conversation_compression_telemetry._log_attempt_record", records.append)
    agent = BackupAgent()
    def simulated_host(*, worker, fence, **kwargs):
        worker(fence)
        if cancel:
            fence.try_cancel_before_commit()
        return kwargs["messages"], "system"
    def summary(agent, messages, system_message, **kwargs):
        guard = aux._CodexStreamGuard(SimpleNamespace(base_url="offline"), 300)
        guard.on_event({"type": "response.reasoning_text.delta", "delta": "PRIVATE"})
        return messages, system_message
    monkeypatch.setattr(cc, "run_compress_context_with_progress_timeout", simulated_host)
    monkeypatch.setattr(cc, "compress_context", summary)
    agent._compress_context([{"role": "user", "content": "DO NOT LOG"}], "system")
    progress = [r for r in records if r.get("event") == "compression_stream_progress"]
    assert len(progress) == 1
    assert progress[0]["cancelled"] is cancel
    assert progress[0]["stream_reasoning_events"] == 1
    assert progress[0]["stream_summary_events"] == 0
    assert progress[0]["stream_first_summary_ms"] is None
    assert progress[0]["idle_timeout_seconds"] == progress[0]["total_ceiling_seconds"] == 120
    assert "PRIVATE" not in json.dumps(records) and "DO NOT LOG" not in json.dumps(records)


def test_stream_observer_reaches_real_protected_provider_daemon():
    from agent.auxiliary_stream_progress import AuxStreamProgress
    from agent.conversation_compression import CompressionCommitFence
    progress = AuxStreamProgress()
    fence = CompressionCommitFence()
    fence.limit_precommit_deadline(10)
    def provider_callback(kwargs):
        guard = aux._CodexStreamGuard(SimpleNamespace(base_url="offline"), 300)
        guard.on_event({"type": "response.output_text.delta", "delta": "provider thread summary"})
        return "done"
    with aux.aux_interrupt_protection(cancel_check=lambda: fence.is_cancelled), aux.aux_stream_observation_hook(progress.observe):
        assert aux._run_protected_sync_provider_call(provider_callback, {}) == "done"
    assert progress.snapshot()["stream_summary_events"] == 1


def test_actual_host_timeout_emits_progress_before_blocked_provider_unwinds(native_gate, monkeypatch):
    import threading
    from tools.thread_context import propagate_context_to_thread
    from agent.context_compressor import _DB_PERSISTED_MARKER
    propagate_context_to_thread(lambda: None)()
    monkeypatch.setattr(config, "load_config", lambda: {"compression": {"native_backup_timeout_seconds": .2}})
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda *a: {})
    records = []
    monkeypatch.setattr("agent.conversation_compression_telemetry._log_attempt_record", records.append)
    release, finished = threading.Event(), threading.Event()
    fences = []
    def blocked_worker(agent, messages, *args, **kwargs):
        fence = kwargs["commit_fence"]
        fences.append(fence)
        try:
            guard = aux._CodexStreamGuard(SimpleNamespace(base_url="offline"), 300)
            with aux.aux_progress_hook(fence.touch_progress):
                guard.on_event({"type": "response.reasoning_text.delta", "delta": "DO NOT LOG"})
            release.wait(2)
        finally:
            finished.set()
        return messages, "system"
    monkeypatch.setattr(cc, "compress_context", blocked_worker)
    try:
        messages = [{"role": "user", "content": "PRIVATE"}]
        result = BackupAgent()._compress_context(messages, "system")
        assert result[0] is messages
        assert not finished.is_set(), "host telemetry must not need worker-finally"
        record = next(r for r in records if r.get("event") == "compression_stream_progress")
        assert record["cancelled"] is True and record["timeout_cause"] == "total_ceiling"
        assert record["stream_reasoning_events"] == 1 and record["stream_summary_events"] == 0
        assert record["native_backup"] is True and record["duration_ms"] < 800
        assert "DO NOT LOG" not in json.dumps(records) and "PRIVATE" not in json.dumps(records)
        assert not fences[0].begin_commit()
    finally:
        release.set()
        assert finished.wait(1)
