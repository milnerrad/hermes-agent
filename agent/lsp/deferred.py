"""LSP diagnostics for a write, delivered after the write's tool result instead of inside it.

``write_file`` / ``patch`` return as soon as the bytes are on disk and verified; the language server
check runs on the LSP loop (``agent.lsp.edit_check``).  Its verdict is queued here under the session
that made the edit (bound by ``tools.file_tools`` around the write) and handed to that session's agent
at its next tool result, or its next user turn when no tool runs again
(``agent.plugin_live_notes``: a new row not yet flushed, so no cached byte moves).

A later edit of the same file takes over an undelivered verdict: the newer check diffs against the
older baseline, so an error the first edit introduced is still reported unless the second fixed it.

A verdict must not be lost when the edit was the turn's last tool call (a one-shot ``-q`` run has no
next turn): checks still running are tracked per session (:func:`track_pending`), and before the turn's
final answer is accepted :func:`settle_for_agent` waits for them (bounded) and returns every verdict
still undelivered, which ``agent.turn_stop_gates`` hands to the model as a new message.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional

_OWNER: ContextVar[Optional[str]] = ContextVar("hermes_lsp_deferred_owner", default=None)
_MAX_OWNERS = 64  # sessions that never drain (a one-shot caller) must not grow this forever
_MAX_PER_OWNER = 32
_LOCK = threading.Lock()
_SETTLED = threading.Condition(_LOCK)
# Warm final settlement retains its five-second ceiling; cold checks carry a bounded allowance.
TURN_END_WAIT = 5.0


@dataclass(frozen=True)
class _Note:
    text: str
    baseline_text: Optional[str]  # pre-edit content the verdict was diffed against


# owner -> {abs path -> note}, oldest owner first
_QUEUED: OrderedDict[str, OrderedDict[str, _Note]] = OrderedDict()
# owner -> tokens of its checks still running (their verdict lands in _QUEUED or nowhere)
_PENDING: dict[str, dict[object, tuple[str, float]]] = {}
_REPORTED: set[object] = set()


def unavailable(path: str) -> str:
    return f"LSP diagnostics unavailable for {path}: check failed, was skipped, or exceeded its bounded wait. The edit is NOT verified clean."


@contextlib.contextmanager
def bind_owner(owner: Optional[str]) -> Iterator[None]:
    """Attribute writes in this block to ``owner`` (the session id, else the task id)."""
    token = _OWNER.set(owner or None)
    try:
        yield
    finally:
        _OWNER.reset(token)


def current_owner() -> Optional[str]:
    return _OWNER.get()


def post(owner: str, path: str, text: str, baseline_text: Optional[str]) -> None:
    """Queue ``text`` (a formatted diagnostics block) for ``owner``.  A note still queued for ``path``
    (an older check finished while a newer one was starting) is kept in front, with its baseline."""
    with _LOCK:
        notes = _QUEUED.pop(owner, None) or OrderedDict()
        older = notes.pop(path, None)
        if older is not None and older.text != text:
            text, baseline_text = f"{older.text}\n\n{text}", older.baseline_text
        notes[path] = _Note(text, baseline_text)
        while len(notes) > _MAX_PER_OWNER:
            notes.popitem(last=False)
        _QUEUED[owner] = notes
        while len(_QUEUED) > _MAX_OWNERS:
            _QUEUED.popitem(last=False)


def take(owner: Optional[str], path: str) -> Optional[_Note]:
    """Withdraw ``owner``'s undelivered note for ``path``: a newer edit of the file supersedes it and
    diffs against its ``baseline_text``.  The newer check carries it and hands it back to
    :func:`restore` only if it ends with no verdict, so the older errors are not lost."""
    if not owner:
        return None
    with _LOCK:
        notes = _QUEUED.get(owner)
        return notes.pop(path, None) if notes else None


def restore(owner: Optional[str], path: str, notes) -> None:
    """Queue ``notes`` (oldest first) withdrawn by :func:`take` again."""
    if owner:
        for note in notes:
            post(owner, path, note.text, note.baseline_text)


def pop_notes(*owners: Optional[str]) -> list[tuple[str, str]]:
    """Every queued ``(path, text)`` for ``owners``, oldest first, and clear them."""
    out: list[tuple[str, str]] = []
    with _LOCK:
        for owner in dict.fromkeys(o for o in owners if o):
            notes = _QUEUED.pop(owner, None)
            if notes:
                out.extend((path, note.text) for path, note in notes.items())
    return out


def track_pending(owner: str, path: str = "edited file", *, timeout: float = TURN_END_WAIT) -> object:
    """Record a check of ``owner``'s still running; pass the token to :func:`settled` once its verdict
    has been posted (or dropped)."""
    token = object()
    with _LOCK:
        _PENDING.setdefault(owner, {})[token] = (path, time.monotonic() + min(15.0, max(0.0, timeout)))
    return token


def settled(owner: str, token: object) -> None:
    with _SETTLED:
        tokens = _PENDING.get(owner)
        if tokens is not None:
            tokens.pop(token, None)
            _REPORTED.discard(token)
            if not tokens:
                del _PENDING[owner]
        _SETTLED.notify_all()


def _owners(agent: Any) -> tuple[Optional[str], ...]:
    owners = (getattr(agent, "session_id", None), getattr(agent, "_current_task_id", None))
    return tuple(o if isinstance(o, str) else None for o in owners)


def settle_for_agent(agent: Any, timeout: Optional[float] = None) -> list[tuple[str, str]]:
    """Wait to the latest owned check deadline (at most 15s), or an explicit caller timeout.
    Drain completed notes atomically; disclose unfinished checks once without forgetting their owner.
    No running checks means no wait. A future real verdict still belongs to this session."""
    owners = [o for o in dict.fromkeys(_owners(agent)) if o]
    if not owners:
        return []
    with _SETTLED:
        def remaining():
            return [(token, entry) for o in owners for token, entry in _PENDING.get(o, {}).items()
                    if token not in _REPORTED]
        deadline = max((entry[1] for _, entry in remaining()), default=time.monotonic())
        wait = max(0.0, deadline - time.monotonic()) if timeout is None else max(0.0, timeout)
        _SETTLED.wait_for(lambda: not remaining(), timeout=min(15.0, wait))
        # Retain ownership for a later real verdict, but never wait/nudge repeatedly for this token.
        unknown = [(path, unavailable(path)) for token, (path, _) in remaining()]
        _REPORTED.update(token for token, _ in remaining())
        out = []
        for owner in owners:
            notes = _QUEUED.pop(owner, None)
            if notes:
                out.extend((path, note.text) for path, note in notes.items())
        return out + unknown


def forward_to_agent(agent: Any) -> None:
    """Move ``agent``'s finished verdicts onto its live-notes queue, which rides the next tool result
    (or the next user turn).  Called right before that tool result is built."""
    notes = pop_notes(*_owners(agent))
    if not notes:
        return
    from agent.plugin_live_notes import queue_plugin_notice
    for path, text in notes:
        queue_plugin_notice(agent, f"lsp:{path}", text)


__all__ = ["TURN_END_WAIT", "bind_owner", "current_owner", "forward_to_agent", "pop_notes", "post", "restore",
           "settle_for_agent", "settled", "take", "track_pending"]
