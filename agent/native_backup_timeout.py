"""Budget policy for Hermes recovery on an otherwise native-compaction request."""
from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar

_NATIVE_BACKUP_ATTEMPT = ContextVar("native_backup_attempt", default=False)


@contextmanager
def native_backup_scope(enabled):
    token = _NATIVE_BACKUP_ATTEMPT.set(bool(enabled))
    try:
        yield
    finally:
        _NATIVE_BACKUP_ATTEMPT.reset(token)


def native_backup_preserves_history():
    return _NATIVE_BACKUP_ATTEMPT.get()

DEFAULT_NATIVE_BACKUP_TIMEOUT_SECONDS = 120.0


class _RejectedNativeView:
    """Recheck every request gate except the flag disabled by a genuine native rejection."""
    codex_responses_native_compaction = True

    def __init__(self, agent):
        self._agent = agent

    def __getattr__(self, name):
        return getattr(self._agent, name)


def native_backup_timeouts(agent, *, force: bool, idle: float, ceiling: float):
    """Return bounded (idle, total), or None for manual / non-native requests.

    A rejection marker is not blanket eligibility: route/model/checkpoint/request policy
    still has to pass. Ordinary local-GPU prompt-reading allowances stay untouched.
    """
    if force or getattr(agent, "api_mode", None) != "codex_responses":
        return None
    from agent.codex_responses_adapter import native_responses_owns_automatic_compaction

    view = _RejectedNativeView(agent) if getattr(agent, "_native_compaction_rejected", False) is True else agent
    if not native_responses_owns_automatic_compaction(view):
        return None
    from hermes_cli.config import load_config
    cfg = load_config().get("compression", {})
    if not isinstance(cfg, dict):
        cfg = {}

    def positive(key, default):
        value = cfg.get(key, default)
        if isinstance(value, bool):
            return default
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) and value > 0 else default

    hard = min(positive("native_backup_timeout_seconds", DEFAULT_NATIVE_BACKUP_TIMEOUT_SECONDS),
               positive("context_total_ceiling_seconds", ceiling), ceiling)
    # Explicitly disabled ordinary idle watchdog does not disable this backup ceiling.
    idle = positive("context_timeout_seconds", idle) if idle > 0 else hard
    return min(idle, hard), hard
