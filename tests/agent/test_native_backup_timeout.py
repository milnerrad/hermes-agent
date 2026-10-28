"""Native-route backup budgets and explicit compression timeout policy (offline)."""

import math

import pytest

from agent import auxiliary_client as aux
from hermes_cli import config
from agent import codex_responses_adapter as adapter
from agent.compression_facade import CompressionFacadeMixin
from types import SimpleNamespace


@pytest.fixture
def native_gate(monkeypatch):
    # Never replace the real shared helper once the scheduling sibling is integrated.
    if hasattr(adapter, "native_responses_owns_automatic_compaction"):
        return
    def gate(agent):
        from agent.native_compaction import native_compaction_context_management
        if getattr(agent, "api_mode", None) != "codex_responses":
            return False
        route = adapter.classify_responses_route(agent)
        return bool(native_compaction_context_management(agent, **route._asdict()))
    monkeypatch.setattr(adapter, "native_responses_owns_automatic_compaction", gate, raising=False)


class BackupAgent(CompressionFacadeMixin):
    provider = "openai-codex"
    base_url = "https://chatgpt.com/backend-api/codex/"
    model = "gpt-5.6-luna"
    api_mode = "codex_responses"
    codex_responses_native_compaction = True
    compression_enabled = True
    session_id = ""
    capabilities = {}
    _cached_system_prompt = "system"

    def __init__(self):
        self.context_compressor = SimpleNamespace(threshold_tokens=255000)

    def _conversation_root_id(self):
        return None

    def _build_system_prompt(self, message):
        return message


def test_eligible_native_backup_uses_absolute_default_budget(native_gate, monkeypatch):
    from agent import compression_facade as facade
    from agent import conversation_compression as cc
    monkeypatch.setattr(config, "load_config", lambda: {"compression": {}})
    seen = {}
    def run(agent, worker, messages, system_message, **kwargs):
        seen.update(kwargs)
        return messages, system_message
    monkeypatch.setattr(facade, "_run_under_progress_timeout", run)
    agent = BackupAgent()
    messages = [{"role": "user", "content": "history"}]
    assert agent._compress_context(messages, "system")[0] is messages
    assert seen["idle_timeout"] == 120
    assert seen["total_ceiling"] == 120
    fence = seen["active_fence"]
    deadline = fence.deadline_monotonic
    assert deadline is not None
    assert fence.extend_deadline(600) is False
    assert fence.deadline_monotonic == deadline


@pytest.mark.parametrize("timeout", [15, 120, 420, "90"])
def test_explicit_user_compression_timeout_is_not_raised(tmp_path, monkeypatch, timeout):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        f"auxiliary:\n  compression:\n    timeout: {timeout}\n", encoding="utf-8"
    )
    assert aux._effective_aux_timeout("compression", None) == float(timeout)


@pytest.mark.parametrize("timeout", [None, False, 0, -1, "oops", math.inf, math.nan])
def test_invalid_or_absent_user_timeout_keeps_reasoning_floor(tmp_path, monkeypatch, timeout):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config, "read_raw_config", lambda: {"auxiliary": {"compression": {"timeout": timeout}}})
    monkeypatch.setattr(aux, "_get_task_timeout", lambda task: 120)
    assert aux._effective_aux_timeout("compression", None) == 300


def test_per_call_override_and_unrelated_tasks_keep_their_budget(monkeypatch):
    monkeypatch.setattr(aux, "_get_task_timeout", lambda task: 20)
    assert aux._effective_aux_timeout("compression", 9) == 9
    assert aux._effective_aux_timeout("title_generation", None) == 20


@pytest.mark.parametrize("scoped,live", [(420, 15), (420, 600), (15, None)])
def test_frozen_session_timeout_is_authoritative_over_live_raw(monkeypatch, scoped, live):
    from agent.auxiliary_task_config import session_auxiliary_scope
    raw = {} if live is None else {"auxiliary": {"compression": {"timeout": live}}}
    def forbidden_live_read():
        raise AssertionError("a frozen session must not read live timeout policy")
    monkeypatch.setattr(config, "read_raw_config", forbidden_live_read)
    with session_auxiliary_scope({"compression": {"timeout": scoped}}):
        assert aux._effective_aux_timeout("compression", None) == scoped
        assert aux._effective_aux_timeout("compression", 9) == 9
    monkeypatch.setattr(config, "read_raw_config", lambda: raw)


def test_empty_session_policy_does_not_inherit_live_explicit_timeout(monkeypatch):
    from agent.auxiliary_task_config import session_auxiliary_scope
    monkeypatch.setattr(config, "read_raw_config", lambda: {"auxiliary": {"compression": {"timeout": 15}}})
    monkeypatch.setattr(aux, "_get_task_timeout", lambda task: 120)
    with session_auxiliary_scope({}):
        assert aux._effective_aux_timeout("compression", None) == 300


def test_effective_ceiling_accepts_legacy_deadline_fence_protocol():
    from agent.conversation_compression_deadline import effective_ceiling
    fence = SimpleNamespace(deadline_monotonic=150)
    assert effective_ceiling(fence, 100, 20) == 130


def test_native_fence_never_relaxes_an_existing_shorter_deadline():
    from agent.conversation_compression import CompressionCommitFence
    from agent.conversation_compression_deadline import effective_ceiling
    import time
    fence = CompressionCommitFence(30)
    before = fence.deadline_monotonic
    fence.limit_precommit_deadline(120)
    fence.set_total_ceiling_seconds(600)
    assert fence.deadline_monotonic == before
    fence.set_precommit_deadline_monotonic(time.monotonic() + 600)
    assert fence.deadline_monotonic == before
    assert effective_ceiling(fence, 600, time.monotonic()) <= 30
