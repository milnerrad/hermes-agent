"""Real host/fence execution with synthetic streams; no provider/client/auth calls."""
import copy
import threading
import time
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent import compression_facade as facade
from agent import conversation_compression as cc
from hermes_cli import config
from tests.agent.test_native_backup_timeout import BackupAgent, native_gate


@pytest.mark.parametrize("kind", ["lifecycle", "whitespace", "reasoning", "summary"])
def test_backup_preserves_history_and_forbids_retry_late_commit(native_gate, monkeypatch, kind):
    from tools.thread_context import propagate_context_to_thread
    from agent.context_compressor import _DB_PERSISTED_MARKER  # prewarm post-return marker mirror
    propagate_context_to_thread(lambda: None)()
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda *a: {})
    monkeypatch.setattr(config, "load_config", lambda: {"compression": {
        "native_backup_timeout_seconds": .18, "context_timeout_seconds": .08,
        "context_total_ceiling_seconds": 5,
    }})
    calls, fences = [], []
    finished = threading.Event()
    def streaming_loop(fence, event):
        guard = aux._CodexStreamGuard(SimpleNamespace(base_url="offline"), 300)
        while not fence.is_cancelled:
            guard.on_event(event)
            time.sleep(.003)

    def worker(agent, messages, system_message, **kwargs):
        fence = kwargs["commit_fence"]
        calls.append(kind)
        fences.append(fence)
        event = {"type": {"lifecycle": "response.in_progress", "whitespace": "response.output_text.delta",
                          "reasoning": "response.reasoning_summary_text.delta", "summary": "response.output_text.delta"}[kind],
                 "delta": {"lifecycle": "", "whitespace": " \n\t", "reasoning": "private reasoning",
                           "summary": "useful text"}[kind]}
        try:
            with aux.aux_progress_hook(fence.touch_progress), aux.aux_stream_deadline(lambda: fence.deadline_monotonic):
                streaming_loop(fence, event)
        except (aux.AuxiliaryExplicitCancellation, TimeoutError):
            pass
        finally:
            finished.set()
        return messages, system_message
    monkeypatch.setattr(cc, "compress_context", worker)
    monkeypatch.setattr(cc, "_retry_compression_on_fallback_chain", lambda **kw: pytest.fail("no timeout fallback calls"))
    agent = BackupAgent()
    # Exceeded-window recovery formerly forced a deterministic discard, even on first timeout.
    agent.context_compressor.context_length = 100
    messages = [{"role": "user", "content": "original private history"}]
    original = copy.deepcopy(messages)
    start = time.monotonic()
    result = agent._compress_context(messages, "system", approx_tokens=200)
    elapsed = time.monotonic() - start
    assert finished.wait(1)
    assert .05 <= elapsed < .8
    assert calls == [kind]
    assert result == (messages, "system") and result[0] is messages
    assert messages == original
    assert fences[0].is_cancelled and not fences[0].begin_commit()
    assert fences[0].extend_deadline(600) is False


@pytest.mark.parametrize("force,provider,api_mode,flag,expected", [
    (True, "openai-codex", "codex_responses", True, (300, 600)),
    (False, "local", "chat_completions", True, (300, 600)),
    (False, "openai-codex", "chat_completions", True, (300, 600)),
    (False, "openai-codex", "codex_responses", False, (300, 600)),
    (False, "openai-codex", "codex_responses", True, (120, 120)),
])
def test_only_eligible_automatic_backup_gets_new_budget(native_gate, monkeypatch, force, provider, api_mode, flag, expected):
    monkeypatch.setattr(config, "load_config", lambda: {"compression": {}})
    monkeypatch.setattr(aux, "_effective_aux_timeout", lambda *a: 300)
    captured = {}
    def run(agent, worker, messages, system_message, **kwargs):
        captured.update(kwargs)
        return messages, system_message
    monkeypatch.setattr(facade, "_run_under_progress_timeout", run)
    agent = BackupAgent()
    agent.provider, agent.api_mode, agent.codex_responses_native_compaction = provider, api_mode, flag
    if provider == "local":
        agent.base_url = "http://localhost:8080"
    agent._compress_context([], "system", force=force)
    assert (captured["idle_timeout"], captured["total_ceiling"]) == expected


@pytest.mark.parametrize("settings,expected", [
    ({"native_backup_timeout_seconds": 45}, (45, 45)),
    ({"native_backup_timeout_seconds": 240}, (240, 240)),
    ({"native_backup_timeout_seconds": 0}, (120, 120)),
    ({"native_backup_timeout_seconds": False}, (120, 120)),
    ({"native_backup_timeout_seconds": "nan"}, (120, 120)),
    ({"context_timeout_seconds": 0}, (120, 120)),
    ({"context_timeout_seconds": 15, "context_total_ceiling_seconds": 30}, (15, 30)),
    ({"native_backup_timeout_seconds": 240, "context_total_ceiling_seconds": 60}, (60, 60)),
])
def test_supported_tuning_never_relaxes_shorter_host_ceiling(native_gate, monkeypatch, settings, expected):
    monkeypatch.setattr(config, "load_config", lambda: {"compression": settings})
    monkeypatch.setattr(aux, "_effective_aux_timeout", lambda *a: 300)
    captured = {}
    def run(agent, worker, messages, system_message, **kwargs):
        captured.update(kwargs)
        return messages, system_message
    monkeypatch.setattr(facade, "_run_under_progress_timeout", run)
    BackupAgent()._compress_context([], "system")
    assert (captured["idle_timeout"], captured["total_ceiling"]) == expected


def test_rejected_native_fallback_keeps_backup_bound(native_gate, monkeypatch):
    from agent import turn_recovery as recovery
    from agent.turn_retry_state import TurnRetryState
    from agent.error_classifier import FailoverReason
    monkeypatch.setattr(config, "load_config", lambda: {"compression": {}})
    monkeypatch.setattr(recovery, "_vlines", lambda *a: None)
    agent = BackupAgent()
    agent.log_prefix = ""
    error = Exception("do not log provider content")
    monkeypatch.setattr("agent.native_compaction.is_native_compaction_rejection", lambda *a: True)
    retry = TurnRetryState()
    assert recovery._recover_format_errors(agent, error, SimpleNamespace(reason=FailoverReason.unknown), retry, [], [])
    assert agent.codex_responses_native_compaction is False
    captured = {}
    def run(agent, worker, messages, system_message, **kwargs):
        captured.update(kwargs)
        return messages, system_message
    monkeypatch.setattr(facade, "_run_under_progress_timeout", run)
    agent._compress_context([], "system")
    assert captured["total_ceiling"] == 120
    # The remembered rejection does not bypass the exact route gate after a route switch.
    agent.provider, agent.api_mode, agent.base_url = "local", "chat_completions", "http://localhost:8080"
    agent._compress_context([], "system")
    assert captured["total_ceiling"] == 600


def test_native_backup_summary_failure_cannot_discard_history(native_gate, monkeypatch):
    from agent.context_compressor import ContextCompressor
    monkeypatch.setattr(config, "load_config", lambda: {"compression": {}})
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **kw: 100000)
    compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
                                   abort_on_summary_failure=False)
    monkeypatch.setattr(compressor, "_generate_summary", lambda *a, **kw: None)
    monkeypatch.setattr(cc, "compress_context", lambda agent, messages, *args, **kw: (compressor.compress(messages), "system"))
    messages = [{"role": "system", "content": "sys"}] + [
        {"role": "user" if i % 2 else "assistant", "content": f"important original turn {i}"} for i in range(1, 8)
    ]
    agent = BackupAgent()
    original = copy.deepcopy(messages)
    result = agent._compress_context(messages, "system")
    assert messages == original and result[0] == original
    assert compressor._last_compress_aborted is True
    assert compressor._last_summary_fallback_used is False
    assert compressor.abort_on_summary_failure is False  # no shared/global policy mutation
