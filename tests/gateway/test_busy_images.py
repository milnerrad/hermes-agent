"""Native busy image delivery, using real cached pixels and real agent control."""
import base64
import contextlib
import copy
import io
import threading
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from run_agent import AIAgent


def bare_agent():
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._executing_tools = True
    agent._execution_thread_id = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent._current_streamed_assistant_text = ""
    agent._strip_think_blocks = lambda text: text
    agent.quiet_mode = True
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    agent.model = "test-vision"
    agent._model_supports_vision = lambda: True
    return agent


@pytest.fixture
def setup_images(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    cache = home / "cache/images"
    cache.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "agent": {"image_input_mode": "native"}, "privacy": {"redact_pii": False},
    })
    runner = GatewayRunner(config=GatewayConfig())
    runner._profile_scope_for_source = lambda source: contextlib.nullcontext()
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    runner._agent_has_active_subagents = lambda agent: False
    runner._effective_busy_input_mode = lambda source: "interrupt"
    runner._busy_skill_turn = AsyncMock(return_value=(False, None))
    runner._draining = False
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner._is_user_authorized_for_source = lambda source: True
    runner._admit_bot_message_for_source = lambda source: True
    runner._effective_busy_text_mode = lambda source: "interrupt"
    adapter = type("Adapter", (), {"_pending_messages": {}})()
    runner._delivery_adapter_for = lambda source: adapter
    agent = bare_agent()
    runner._session_state("key").turn.agent = agent
    paths = []
    for index, color in enumerate(("red", "blue")):
        path = cache / f"fixture-{index}.png"
        Image.new("RGB", (3, 2), color).save(path)
        paths.append(str(path))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="sender", profile="profile")
    event = MessageEvent(text="use these instead", message_type=MessageType.PHOTO, source=source,
                         message_id="photo-message", media_urls=paths, media_types=["image/png"] * 2,
                         reply_to_message_id="earlier", reply_to_text="the old reference")
    return runner, agent, event, cache


def pixel_colors(content):
    urls = [part["image_url"]["url"] for part in content if part.get("type") == "image_url"]
    return [Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).getpixel((0, 0)) for url in urls]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["normal", "priority"])
async def test_human_photos_reach_checkpoint_not_queue(setup_images, route):
    from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
    runner, agent, event, _ = setup_images
    if route == "normal":
        assert await runner._handle_active_session_busy_message(event, "key")
        assert not runner._delivery_adapter_for(event.source)._pending_messages
        assert not runner._session_state("key").conversation.queued_events
    else:
        runner._hm_merge_pending_for_source = lambda *args, **kwargs: pytest.fail("photo unexpectedly queued")
        await runner._hm_handle_running_session_message(event, event.source, "key")
    messages = [{"role": "assistant", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "probe", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "t", "content": "done"}]
    prefix = copy.deepcopy(messages)
    apply_pending_steer_to_tool_results(agent, messages, 1)
    assert messages[:2] == prefix
    row = messages[-1]
    assert row["role"] == "user"
    assert pixel_colors(row["content"]) == [(255, 0, 0), (0, 0, 255)]
    text = "\n".join(p.get("text", "") for p in row["content"])
    assert event.text in text and "the old reference" in text and "sender" in text
    assert agent._drain_pending_steer() is None

@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["tools", "request"])
async def test_real_codex_request_contains_fixture_pixels(setup_images, phase):
    import json
    import httpx
    from openai import OpenAI
    from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
    from agent.conversation_loop import _apply_active_turn_redirect
    from agent.transports.codex import ResponsesApiTransport

    runner, agent, event, _ = setup_images
    agent._executing_tools = phase == "tools"
    agent._model_request_active.set()
    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
    assert outcome.redirected
    messages = [{"role": "user", "content": "original"},
                {"role": "assistant", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "probe", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "t", "content": "done"}]
    prefix = copy.deepcopy(messages)
    if phase == "tools":
        apply_pending_steer_to_tool_results(agent, messages, 1)
    else:
        _apply_active_turn_redirect(agent, messages, agent._drain_pending_redirect())
    assert messages[:3] == prefix
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "resp_offline", "object": "response", "created_at": 0,
                                        "model": "test-vision", "status": "completed", "output": []})

    kwargs = ResponsesApiTransport().build_kwargs("test-vision", messages, session_id="image-session")
    with OpenAI(api_key="offline-test", base_url="https://provider.invalid/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(respond)), max_retries=0) as client:
        client.responses.create(**kwargs)
    parts = [part for item in captured[0]["input"] if isinstance(item.get("content"), list)
             for part in item["content"]]
    urls = [part["image_url"] for part in parts if part.get("type") == "input_image"]
    colors = [Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).getpixel((0, 0)) for url in urls]
    assert colors == [(255, 0, 0), (0, 0, 255)], "actual SDK request must contain the pixels"


@pytest.mark.asyncio
async def test_late_image_steer_recovers_original_events(setup_images):
    runner, agent, event, _ = setup_images
    await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
    leftover = agent._drain_leftover_steer()
    adapter = type("Adapter", (), {"_pending_messages": {}})()
    pending_event, pending = runner._merge_leftover_steer(
        {"pending_steer": leftover}, adapter, event.source, "key", None, None,
    )
    assert pending_event is event, "late delivery must recover attachments and original identity"
    assert pending is None
    assert agent._drain_leftover_steer() is None

@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["tools", "request"])
async def test_ordered_captionless_burst_with_text_and_requeue(setup_images, phase):
    from dataclasses import replace
    from agent.prompt_builder import steer_user_row
    from agent.agent_runtime_helpers import _requeue_pending_steer
    runner, agent, event, _ = setup_images
    agent._executing_tools = phase == "tools"
    agent._model_request_active.set()
    first = replace(event, text="", media_urls=event.media_urls[:1], media_types=event.media_types[:1])
    second = replace(event, text="blue second", message_id="second", media_urls=event.media_urls[1:], media_types=event.media_types[1:])
    assert (await runner._resolve_busy_steer_or_redirect(first, "key", "interrupt", agent)).redirected
    agent.redirect("between images")
    assert (await runner._resolve_busy_steer_or_redirect(second, "key", "interrupt", agent)).redirected
    drain = agent._drain_pending_steer if phase == "tools" else agent._drain_pending_redirect
    payload = drain()
    if phase == "tools":
        _requeue_pending_steer(agent, payload)
        payload = drain()
    row = steer_user_row(payload)
    assert isinstance(row["content"], list), "burst must retain native blocks"
    assert pixel_colors(row["content"]) == [(255, 0, 0), (0, 0, 255)]
    parts = row["content"]
    image_positions = [i for i, part in enumerate(parts) if part.get("type") == "image_url"]
    between_position = next(i for i, part in enumerate(parts) if "between images" in part.get("text", ""))
    assert image_positions[0] < between_position < image_positions[1]
    assert drain() is None


def _alter_media_fixture(condition, event, cache):
    def corrupt():
        path = cache / "corrupt.png"
        path.write_bytes(b"not pixels")
        return path

    def outside():
        path = cache.parent / "private.png"
        Image.new("RGB", (1, 1), "green").save(path)
        return path

    def symlink():
        path = cache / "link.png"
        path.symlink_to(outside())
        return path

    def hardlink():
        import os
        path = cache / "linked.png"
        os.link(outside(), path)
        return path

    def cache_symlink():
        parent = cache.parent
        moved = parent.with_name("moved-cache")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
        return event.media_urls[1]

    def traversal():
        outside()
        return str(cache) + "/../private.png"

    bad_paths = {"missing": lambda: cache / "missing.png", "corrupt": corrupt,
                 "outside": outside, "symlink": symlink, "traversal": traversal,
                 "hardlink": hardlink, "cache-ancestor-symlink": cache_symlink}
    if condition in bad_paths:
        event.media_urls[1] = str(bad_paths[condition]())
    if condition in ("mixed", "mixed-voice", "voice-with-image"):
        audio = condition != "mixed"
        event.media_urls.append(str(cache / ("voice.ogg" if audio else "movie.mp4")))
        event.media_types.append("audio/ogg" if audio else "video/mp4")
    if condition == "voice-with-image":
        event.message_type = MessageType.VOICE
    if condition == "nonimage":
        event.message_type = MessageType.DOCUMENT
        event.media_types = ["application/octet-stream"] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["normal", "priority"])
@pytest.mark.parametrize("condition", ["queue", "subagents", "compression", "internal", "text-model", "runtime", "mixed", "nonimage", "missing", "corrupt", "outside", "symlink", "traversal", "text-mode", "old-runtime", "mixed-voice", "cache-ancestor-symlink", "hardlink", "voice-with-image"])
async def test_unsupported_images_queue_intact(setup_images, monkeypatch, route, condition):
    runner, agent, event, cache = setup_images
    mode = "queue" if condition == "queue" else "interrupt"
    alterations = {
        "subagents": lambda: setattr(runner, "_agent_has_active_subagents", lambda a: True),
        "compression": lambda: setattr(runner, "_session_has_compression_in_flight", AsyncMock(return_value=True)),
        "internal": lambda: setattr(event, "internal", True),
        "text-model": lambda: setattr(agent, "_model_supports_vision", lambda: False),
        "runtime": lambda: setattr(agent, "api_mode", "codex_app_server"),
        "text-mode": lambda: monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"agent": {"image_input_mode": "tool"}}),
    }
    if condition in alterations:
        alterations[condition]()
    if condition == "old-runtime":
        agent = object()
        runner._session_state("key").turn.agent = agent
    _alter_media_fixture(condition, event, cache)
    original = (event.text, list(event.media_urls), list(event.media_types), event.source, event.reply_to_message_id)
    runner._effective_busy_input_mode = lambda source: mode
    queued = []
    if route == "priority":
        runner._hm_merge_pending_for_source = lambda source, key, value, **kw: queued.append(value)
        runner._queue_or_replace_pending_event = lambda key, value: queued.append(value)
        await runner._hm_handle_running_session_message(event, event.source, "key")
        assert queued == [event]
    else:
        outcome = await runner._resolve_busy_steer_or_redirect(event, "key", mode, agent)
        assert not outcome.redirected and not outcome.steered
        assert outcome.effective_mode == "queue"
    if isinstance(agent, AIAgent):
        assert agent._drain_pending_steer() is None
        assert agent._drain_pending_redirect() is None
        assert not agent._interrupt_requested
    assert original == (event.text, event.media_urls, event.media_types, event.source, event.reply_to_message_id)


@pytest.mark.asyncio
async def test_priority_text_reply_inheriting_image_is_native(setup_images):
    runner, agent, event, _ = setup_images
    event.message_type = MessageType.TEXT  # reply-image inheritance is materialized before busy routing
    runner._hm_merge_pending_for_source = lambda *a, **kw: pytest.fail("inherited image queued")
    await runner._hm_handle_running_session_message(event, event.source, "key")
    from agent.prompt_builder import steer_user_row
    assert pixel_colors(steer_user_row(agent._drain_pending_steer())["content"]) == [(255, 0, 0), (0, 0, 255)]


@pytest.mark.asyncio
async def test_image_steer_mode_does_not_abort_model_request(setup_images):
    runner, agent, event, _ = setup_images
    agent._executing_tools = False
    agent._model_request_active.set()
    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "steer", agent)
    assert outcome.steered
    assert not agent._interrupt_requested
    assert agent._drain_pending_redirect() is None
    from agent.prompt_builder import steer_user_row
    assert len(pixel_colors(steer_user_row(agent._drain_pending_steer())["content"])) == 2

@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["tools", "request", "pre-api"])
async def test_model_losing_vision_returns_images_to_fallback(setup_images, phase):
    from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
    from agent.conversation_loop import _apply_active_turn_redirect
    from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result
    runner, agent, event, _ = setup_images
    agent._executing_tools = phase != "request"
    agent._model_request_active.set()
    await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
    agent._model_supports_vision = lambda: False
    messages = [{"role": "tool", "tool_call_id": "t", "content": "done"}]
    before = copy.deepcopy(messages)
    if phase == "tools":
        apply_pending_steer_to_tool_results(agent, messages, 1)
    elif phase == "pre-api":
        _inject_steer_after_newest_tool_result(agent, messages, agent._drain_pending_steer())
    else:
        _apply_active_turn_redirect(agent, messages, agent._drain_pending_redirect())
    assert messages == before, "unsupported model must receive no native blocks or caption-only correction"
    leftover = agent._drain_leftover_steer()
    pending_event, pending = runner._merge_leftover_steer(
        {"pending_steer": leftover}, None, event.source, "key", None, None,
    )
    assert pending_event is event and pending is None


@pytest.mark.asyncio
async def test_profile_session_isolation_real_runtime_scopes(setup_images, tmp_path):
    import asyncio
    from dataclasses import replace
    from agent.prompt_builder import steer_user_row
    from gateway.run import _profile_runtime_scope
    runner, agent, event, cache = setup_images
    other_home = tmp_path / "other-profile"
    other_cache = other_home / "cache/images"
    other_cache.mkdir(parents=True)
    other_path = other_cache / "other.png"
    Image.new("RGB", (2, 2), "green").save(other_path)
    homes = {"profile": cache.parent.parent, "other": other_home}
    runner._profile_scope_for_source = lambda source: _profile_runtime_scope(homes[source.profile])
    other_agent = bare_agent()
    runner._session_state("other-key").turn.agent = other_agent
    other_source = replace(event.source, profile="other", chat_id="other-chat", user_id="other-sender")
    other = replace(event, source=other_source, text="other caption", media_urls=[str(other_path)], media_types=["image/png"])
    outcomes = await asyncio.gather(
        runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent),
        runner._resolve_busy_steer_or_redirect(other, "other-key", "interrupt", other_agent),
    )
    assert all(outcome.redirected for outcome in outcomes)
    first = steer_user_row(agent._drain_pending_steer())["content"]
    second = steer_user_row(other_agent._drain_pending_steer())["content"]
    assert pixel_colors(first) == [(255, 0, 0), (0, 0, 255)]
    assert pixel_colors(second) == [(0, 128, 0)]
    assert "other-sender" not in str(first) and "other caption" not in str(first)
    assert "use these instead" not in str(second) and '"profile": "other"' in str(second)
    # Even valid pixels from the neighboring profile are forbidden, not silently adopted.
    wrong = replace(other, media_urls=event.media_urls, media_types=event.media_types)
    assert not (await runner._resolve_busy_steer_or_redirect(wrong, "other-key", "interrupt", other_agent)).redirected


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["tools", "request"])
async def test_explicit_stop_stays_stop(setup_images, phase):
    runner, agent, event, _ = setup_images
    agent._executing_tools = phase == "tools"
    agent._model_request_active.set()
    assert (await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)).redirected
    agent.interrupt(hard_cancel=True)
    assert agent._interrupt_requested
    assert agent._drain_pending_redirect() is None
    assert not agent.redirect_images("not an image envelope")
    # Explicit Stop supersedes pending corrections just as it does text. It must
    # not resurrect a cancelled image turn after the active request stops.
    agent.clear_interrupt(hard_cancel=True)
    assert agent._drain_leftover_steer() is None


@pytest.mark.asyncio
async def test_pre_api_checkpoint_preserves_pixels_and_original_tool_row(setup_images):
    from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result
    runner, agent, event, _ = setup_images
    await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
    rows = [{"role": "tool", "tool_call_id": "t", "content": "original tool bytes"}]
    before = copy.deepcopy(rows)
    _inject_steer_after_newest_tool_result(agent, rows, agent._drain_pending_steer())
    assert rows[:1] == before
    assert pixel_colors(rows[-1]["content"]) == [(255, 0, 0), (0, 0, 255)]


@pytest.mark.asyncio
async def test_late_images_do_not_reorder_unrelated_fifo(setup_images):
    from dataclasses import replace
    runner, agent, event, _ = setup_images
    existing = replace(event, text="unrelated", media_urls=[], media_types=[], message_type=MessageType.TEXT)
    later = replace(existing, text="later unrelated")
    runner._session_state("key").conversation.queued_events.append(later)
    await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
    leftover = agent._drain_leftover_steer()
    pending_event, pending = runner._merge_leftover_steer(
        {"pending_steer": leftover}, None, event.source, "key", existing, None,
    )
    assert pending_event is existing and pending is None
    assert runner._session_state("key").conversation.queued_events == [later, event]

@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["agent-replaced", "subagents-started", "compression-started"])
async def test_image_preparation_cannot_deliver_into_stale_or_protected_turn(setup_images, monkeypatch, change):
    import gateway.busy_images as images
    runner, agent, event, _ = setup_images
    prepare = images.prepare_image_steer

    def while_preparing(*args):
        payload = prepare(*args)
        if change == "agent-replaced":
            runner._session_state("key").turn.agent = bare_agent()
        elif change == "subagents-started":
            runner._agent_has_active_subagents = lambda a: True
        else:
            runner._session_has_compression_in_flight = AsyncMock(return_value=True)
        return payload

    monkeypatch.setattr(images, "prepare_image_steer", while_preparing)
    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
    assert not outcome.redirected and outcome.effective_mode == "queue"
    assert agent._drain_pending_steer() is None
    assert agent._drain_pending_redirect() is None


@pytest.mark.asyncio
async def test_unauthorized_image_never_prepared(setup_images, monkeypatch):
    runner, agent, event, _ = setup_images
    runner._is_user_authorized_for_source = lambda source: False
    monkeypatch.setattr("gateway.busy_images._cached_image", lambda *args: pytest.fail("unauthorized image was read"))
    assert await runner._handle_active_session_busy_message(event, "key")
    assert agent._drain_pending_steer() is None
    assert not runner._delivery_adapter_for(event.source)._pending_messages


@pytest.mark.asyncio
async def test_failed_turn_restores_unconsumed_image_redirect(setup_images):
    runner, agent, event, _ = setup_images
    agent._executing_tools = False
    agent._model_request_active.set()
    assert (await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)).redirected
    # Fatal request failure can finalize before begin_iteration applies the correction.
    leftover = agent._drain_leftover_steer()
    pending_event, pending = runner._merge_leftover_steer(
        {"pending_steer": leftover}, None, event.source, "key", None, None,
    )
    assert pending_event is event and pending is None
    assert agent._drain_pending_redirect() is None
    assert agent._drain_leftover_steer() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MessageType.VOICE, MessageType.AUDIO])
async def test_audio_only_retains_legacy_busy_behavior(setup_images, monkeypatch, kind):
    runner, agent, event, cache = setup_images
    event.message_type = kind
    event.media_urls = [str(cache / "voice.ogg")]
    event.media_types = ["audio/ogg"]
    monkeypatch.setattr("gateway.busy_images.prepare_image_steer", lambda *a: pytest.fail("audio misrouted as image"))
    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
    assert outcome.effective_mode == "interrupt" and not outcome.redirected

@pytest.mark.asyncio
async def test_declined_redirect_does_not_pollute_original_prompt(setup_images):
    from types import SimpleNamespace
    from agent.turn_iteration_prep import begin_iteration
    runner, agent, event, _ = setup_images
    agent._executing_tools = False
    agent._model_request_active.set()
    await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
    agent._model_supports_vision = lambda: False
    agent._checkpoint_mgr = SimpleNamespace(new_turn=lambda: None)
    agent._persist_session = lambda *args: None
    messages = [{"role": "user", "content": "original prompt"}]
    state = begin_iteration(agent, messages=messages, conversation_history=[],
                            original_user_message="original prompt", api_call_count=0,
                            interrupted=False, _turn_exit_reason=None)
    assert state.original_user_message == "original prompt"
    assert messages == [{"role": "user", "content": "original prompt"}]

@pytest.mark.asyncio
async def test_real_telegram_reply_image_inheritance_reaches_busy_agent(setup_images):
    from types import SimpleNamespace
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from agent.prompt_builder import steer_user_row
    runner, agent, event, _ = setup_images
    path = event.media_urls[0]
    event.media_urls, event.media_types = [], []
    event.message_type = MessageType.TEXT
    adapter = object.__new__(TelegramAdapter)
    cached = SimpleNamespace(path=path, media_type="image/png", kind="image", display_name="replied.png")
    adapter._download_observed_media = AsyncMock(return_value=("ok", cached))
    await adapter._cache_replied_media(SimpleNamespace(reply_to_message=object()), event)
    assert event.message_type == MessageType.PHOTO and event.media_urls == [path]
    await runner._hm_handle_running_session_message(event, event.source, "key")
    row = steer_user_row(agent._drain_pending_steer())
    assert pixel_colors(row["content"]) == [(255, 0, 0)]
    assert "the old reference" in str(row["content"]) and "Replied-to image" in str(row["content"])


@pytest.mark.asyncio
async def test_concurrent_burst_keeps_arrival_order_despite_slow_first_decode(setup_images, monkeypatch):
    import asyncio
    from dataclasses import replace
    from agent.prompt_builder import steer_user_row
    runner, agent, event, _ = setup_images
    first = replace(event, text="first", media_urls=event.media_urls[:1], media_types=event.media_types[:1])
    second = replace(event, text="second", media_urls=event.media_urls[1:], media_types=event.media_types[1:])
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    entered = []

    async def controlled_decode(fn, *args):
        entered.append(args[-1].text)
        if args[-1].text == "first":
            first_started.set()
            await release_first.wait()
        return fn(*args)

    monkeypatch.setattr(asyncio, "to_thread", controlled_decode)
    a = asyncio.create_task(runner._resolve_busy_steer_or_redirect(first, "key", "interrupt", agent))
    await first_started.wait()
    b = asyncio.create_task(runner._resolve_busy_steer_or_redirect(second, "key", "interrupt", agent))
    try:
        await asyncio.sleep(0)  # schedule the second arrival, not a wall-clock delay
        assert entered == ["first"], "later arrivals must not overtake the pending image decode"
    finally:
        release_first.set()
        outcomes = await asyncio.gather(a, b)
    assert all(outcome.redirected for outcome in outcomes)
    content = steer_user_row(agent._drain_pending_steer())["content"]
    assert pixel_colors(content) == [(255, 0, 0), (0, 0, 255)]
