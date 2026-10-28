"""Content-free Codex stream observations; liveness is not a usable summary."""
from __future__ import annotations

import threading
import time


class AuxStreamProgress:
    """Only fixed counter keys, byte counts and monotonic durations survive each event."""

    def __init__(self):
        self.started_at = time.monotonic()
        self._lock = threading.Lock()
        self._stats = {"stream_events": 0, "stream_completed": False,
                       "stream_first_event_ms": None, "stream_first_reasoning_ms": None,
                       "stream_first_summary_ms": None}
        for category in ("reasoning", "whitespace", "summary"):
            self._stats[f"stream_{category}_events"] = 0
            self._stats[f"stream_{category}_bytes"] = 0

    def observe(self, event):
        def field(obj, key):
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
        kind, delta = field(event, "type"), field(event, "delta")
        category = None
        if isinstance(delta, str) and delta:
            if kind == "response.output_text.delta":
                category = "summary" if delta.strip() else "whitespace"
            elif kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
                category = "reasoning"
        size = len(delta.encode("utf-8")) if category and isinstance(delta, str) else 0
        elapsed = max(0, int((time.monotonic() - self.started_at) * 1000))
        with self._lock:
            self._stats["stream_events"] += 1
            if self._stats["stream_first_event_ms"] is None:
                self._stats["stream_first_event_ms"] = elapsed
            if kind == "response.completed":
                self._stats["stream_completed"] = True
            if category:
                self._stats[f"stream_{category}_events"] += 1
                self._stats[f"stream_{category}_bytes"] += size
                first = f"stream_first_{category}_ms"
                if first in self._stats and self._stats[first] is None:
                    self._stats[first] = elapsed

    def snapshot(self):
        with self._lock:
            return dict(self._stats)
