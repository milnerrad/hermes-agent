"""The recursion cap must defer, not overwrite, already-admitted FIFO events."""
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.turn_context import TurnContext
from gateway.wake import admit_internal_event


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _setup():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = _Adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._draining = False
    runner.config = GatewayConfig()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    key = build_session_key(source)
    result = {"final_response": "previous", "messages": []}
    ctx = TurnContext(source=source, session_key=key, session_id="cap-fifo", history=[],
                      _interrupt_depth=runner._MAX_INTERRUPT_DEPTH, result_holder=[result])
    adapter._active_sessions[key] = asyncio.Event()
    adapter._active_sessions[key].set()
    return runner, adapter, source, key, ctx, result


@pytest.mark.asyncio
async def test_cap_preserves_three_text_events_as_distinct_fifo_turns():
    runner, adapter, source, key, ctx, result = _setup()
    events = [MessageEvent(text=name, source=source, message_id=name) for name in "ABC"]
    for event in events:
        runner._enqueue_fifo(key, event, adapter)

    event, pending = await runner._run_agent_drain_pending(result, adapter, source, key)
    assert event is events[0]
    assert adapter._pending_messages[key] is events[1]
    returned = await runner._run_agent_queued_followup(ctx, adapter, pending, event, "previous", result, None)
    assert returned is result
    assert not adapter._active_sessions[key].is_set()

    # Each pop models a fresh outer turn after the capped recursive chain unwinds.
    consumed = []
    for _ in range(4):
        event, text = await runner._run_agent_drain_pending(result, adapter, source, key)
        if event is None:
            assert text is None
            break
        consumed.append(event)
        assert text == event.message_id
    assert [event.message_id for event in consumed] == ["A", "B", "C"]
    assert all(actual is original for actual, original in zip(consumed, events, strict=True))
    assert runner._queue_depth(key, adapter=adapter) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("kind", [MessageType.TEXT, MessageType.PHOTO, MessageType.DOCUMENT])
async def test_cap_keeps_event_identity_and_attachments(count, kind):
    runner, adapter, source, key, ctx, result = _setup()
    events = [MessageEvent(
        text=f"caption-{i}", message_type=kind, source=source, message_id=f"media-{i}",
        media_urls=[f"/fixture/{i}"], media_types=["image/png"], media_text_inlined=[False],
        reply_to_message_id=f"reply-{i}", channel_prompt=f"pin-{i}",
        internal=i % 2 == 0, allow_gateway_control=False,
        metadata={"delegation_id": f"task-{i}", "notification_category": "diagnostic"},
    ) for i in range(count)]
    for event in events:
        runner._enqueue_fifo(key, event, adapter)
    for _ in range(2):
        event, text = await runner._run_agent_drain_pending(result, adapter, source, key)
        await runner._run_agent_queued_followup(ctx, adapter, text, event, "previous", result, None)
    for i, expected in enumerate(events):
        event, text = await runner._run_agent_drain_pending(result, adapter, source, key)
        assert event is expected
        assert text == f"caption-{i}"
        assert event.source is source
        assert event.media_urls == [f"/fixture/{i}"]
        assert event.media_types == ["image/png"]
        assert event.media_text_inlined == [False]
        assert event.reply_to_message_id == f"reply-{i}"
        assert event.channel_prompt == f"pin-{i}"
        assert event.internal is (i % 2 == 0)
        assert event.allow_gateway_control is False
        assert event.metadata == {"delegation_id": f"task-{i}", "notification_category": "diagnostic"}
    assert runner._queue_depth(key, adapter=adapter) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("result_field", ["interrupt_message", "pending_steer"])
async def test_cap_preserves_eventless_queue_message_fallback(result_field):
    runner, adapter, source, key, ctx, result = _setup()
    result.update({"interrupted": True, result_field: "eventless follow-up"})
    event, text = await runner._run_agent_drain_pending(result, adapter, source, key)
    assert event is None and text == "eventless follow-up"
    queued = []
    adapter.queue_message = lambda session_key, pending: queued.append((session_key, pending))
    await runner._run_agent_queued_followup(ctx, adapter, text, event, "previous", result, None)
    assert queued == [(key, "eventless follow-up")]


@pytest.mark.asyncio
async def test_leftover_steer_at_cap_does_not_displace_queued_events():
    runner, adapter, source, key, ctx, result = _setup()
    events = [MessageEvent(text=name, source=source, message_id=name) for name in "ABC"]
    for event in events:
        runner._enqueue_fifo(key, event, adapter)
    result["pending_steer"] = "late steer"
    event, text = await runner._run_agent_drain_pending(result, adapter, source, key)
    assert event is None and text == "late steer"
    queued = []
    adapter.queue_message = lambda session_key, pending: queued.append((session_key, pending))
    await runner._run_agent_queued_followup(ctx, adapter, text, event, "previous", result, None)
    assert queued == [(key, "late steer")]
    result.pop("pending_steer")
    for expected in events:
        event, _ = await runner._run_agent_drain_pending(result, adapter, source, key)
        assert event is expected
    assert runner._queue_depth(key, adapter=adapter) == 0


@pytest.mark.asyncio
async def test_below_cap_runs_followup_and_keeps_mid_handoff_arrival_last(monkeypatch):
    runner, adapter, source, key, ctx, result = _setup()
    ctx._interrupt_depth = runner._MAX_INTERRUPT_DEPTH - 1
    events = [MessageEvent(text=name, source=source, message_id=name) for name in "ABCD"]
    for event in events[:3]:
        runner._enqueue_fifo(key, event, adapter)
    prepared, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def prepare(*, event, **kwargs):
        assert event is events[0]
        prepared.set()
        await release.wait()
        return event.text

    async def no_io(*args, **kwargs):
        return None

    async def model(**kwargs):
        seen.append(kwargs)
        return {"final_response": "A-done", "messages": []}

    monkeypatch.setattr(runner, "_prepare_profile_scoped_inbound_message_text", prepare)
    monkeypatch.setattr(runner, "_persist_prompt_pins", no_io)
    monkeypatch.setattr(runner, "_refresh_agent_cache_message_count", no_io)
    monkeypatch.setattr(runner, "_run_agent_deliver_first_response", no_io)
    monkeypatch.setattr(runner, "_run_agent", model)
    event, text = await runner._run_agent_drain_pending(result, adapter, source, key)
    task = asyncio.create_task(runner._run_agent_queued_followup(ctx, adapter, text, event, "previous", result, None))
    try:
        await asyncio.wait_for(prepared.wait(), 2)
        runner._enqueue_fifo(key, events[3], adapter)
        release.set()
        returned = await asyncio.wait_for(task, 2)
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert returned["final_response"] == "A-done"
    assert len(seen) == 1
    assert seen[0]["message"] == "A"
    assert seen[0]["inbound_message_id"] == "A"
    assert seen[0]["source"] is source
    assert seen[0]["_interrupt_depth"] == runner._MAX_INTERRUPT_DEPTH
    for expected in events[1:]:
        event, _ = await runner._run_agent_drain_pending(result, adapter, source, key)
        assert event is expected
    assert runner._queue_depth(key, adapter=adapter) == 0


@pytest.mark.asyncio
async def test_real_admission_and_adapter_handoff_consume_capped_completions_once():
    runner, adapter, source, key, ctx, result = _setup()
    # Let handle_message own the guard and real background-task handoff.
    adapter._active_sessions.clear()
    started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    initial = MessageEvent(text="initial", source=source, message_id="initial")
    events = [MessageEvent(text=f"completed {name}", source=source, internal=True,
                          message_id=f"async-delegation:{name}",
                          metadata={"delegation_id": name, "notification_category": "result"})
              for name in "ABC"]
    consumed = []

    async def handler(event):
        consumed.append(event)
        if event is initial:
            started.set()
            await release.wait()
        pending_event, text = await runner._run_agent_drain_pending(result, adapter, source, key)
        if pending_event is not None:
            await runner._run_agent_queued_followup(ctx, adapter, text, pending_event, "previous", result, None)
        if event is events[-1]:
            finished.set()

    adapter.set_message_handler(handler)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    try:
        await adapter.handle_message(initial)
        await asyncio.wait_for(started.wait(), 2)
        for event in events:
            await admit_internal_event(adapter, event)
            assert event._gateway_accepted is True
        release.set()
        await asyncio.wait_for(finished.wait(), 2)
        while adapter._background_tasks:
            await asyncio.gather(*list(adapter._background_tasks))
            await asyncio.sleep(0)
        assert len(consumed) == 4
        assert all(actual is original for actual, original in zip(consumed, [initial, *events], strict=True))
        for name, event in zip("ABC", consumed[1:], strict=True):
            assert event.message_id == f"async-delegation:{name}"
            assert event.internal and event.source is source
            assert event.metadata == {"delegation_id": name, "notification_category": "result"}
        assert runner._queue_depth(key, adapter=adapter) == 0
        assert key not in adapter._active_sessions
    finally:
        release.set()
        await adapter.cancel_background_tasks()
