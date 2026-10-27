"""Recovered busy-image inputs execute exactly once through the real follow-up consumer."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway import test_busy_images as image_fixtures

setup_images = image_fixtures.setup_images


@pytest.mark.asyncio
@pytest.mark.parametrize("caption", ["use these instead", ""])
async def test_unconsumed_image_drain_can_execute_followup(setup_images, caption):
    runner, agent, event, _ = setup_images
    event.text = caption
    assert (await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)).redirected
    result = {"final_response": "finished first", "messages": [],
              "pending_steer": agent._drain_leftover_steer()}
    adapter = runner._delivery_adapter_for(event.source)
    from gateway.platforms.base import BasePlatformAdapter
    adapter.get_pending_message = BasePlatformAdapter.get_pending_message.__get__(adapter)
    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, event.source, "key")
    assert pending_event is event
    ctx = SimpleNamespace(source=event.source, session_id="session", session_key="key",
                          run_generation=1, _interrupt_depth=0, history=[],
                          _status_thread_metadata={}, result_holder=[result],
                          context_prompt="", channel_prompt=None,
                          mute_notification_reply=False)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="photo prepared")
    runner._session_key_for_source = lambda source: "key"
    runner._pinned_channel_inputs = lambda *args, **kwargs: (None, event.source)
    runner._persist_prompt_pins = AsyncMock()
    runner._intake_adapter_for = lambda source: None
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._run_agent = AsyncMock(return_value={"final_response": "saw original pixels", "messages": []})
    followup = await runner._run_agent_queued_followup(ctx, adapter, pending, pending_event, result, result, None)
    assert followup["final_response"] == "saw original pixels"
    assert runner._run_agent.await_count == 1


@pytest.mark.asyncio
async def test_unconsumed_request_burst_recovers_only_original_events(setup_images):
    runner, agent, first, _ = setup_images
    second = replace(first, text="second photo", message_id="second-photo")
    agent._executing_tools = False
    agent._model_request_active.set()
    assert (await runner._resolve_busy_steer_or_redirect(first, "key", "interrupt", agent)).redirected
    assert (await runner._resolve_busy_steer_or_redirect(second, "key", "interrupt", agent)).redirected
    result = {"final_response": "fatal request error", "failed": True,
              "pending_steer": agent._drain_leftover_steer()}
    adapter = runner._delivery_adapter_for(first.source)
    from gateway.platforms.base import BasePlatformAdapter
    adapter.get_pending_message = BasePlatformAdapter.get_pending_message.__get__(adapter)
    pending_event, _ = await runner._run_agent_drain_pending(result, adapter, first.source, "key")
    recovered = [pending_event]
    next_up = adapter._pending_messages.get("key")
    if next_up is not None:
        recovered.append(next_up)
    recovered.extend(runner._session_state("key").conversation.queued_events)
    assert len(recovered) == 2, [(item.message_id, item.text) for item in recovered]
    assert recovered[0] is first and recovered[1] is second


@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", [False, True])
async def test_render_metadata_is_not_input_but_identical_authored_text_survives(setup_images, exhausted):
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.event import MessageEvent

    runner, agent, first, _ = setup_images
    second = replace(first, text="second photo", message_id="second-photo")
    authored = "[Additional user correction]"
    agent._executing_tools = False
    agent._model_request_active.set()
    assert (await runner._resolve_busy_steer_or_redirect(first, "key", "interrupt", agent)).redirected
    assert agent.redirect(authored)
    assert (await runner._resolve_busy_steer_or_redirect(second, "key", "interrupt", agent)).redirected
    leftover = agent._drain_leftover_steer()
    assert any(authored in part.get("text", "") for part in leftover.content())
    result = {"final_response": "finished", "pending_steer": leftover}
    adapter = runner._delivery_adapter_for(first.source)
    adapter.get_pending_message = BasePlatformAdapter.get_pending_message.__get__(adapter)
    if exhausted:
        anchor = MessageEvent(text="already queued", source=first.source, message_id="anchor")
        runner._defer_past_exhaustion_reset(result, adapter, first.source, "key", anchor)
        assert adapter._pending_messages["key"] is anchor
        recovered = list(runner._session_state("key").conversation.queued_events)
    else:
        event, _ = await runner._run_agent_drain_pending(result, adapter, first.source, "key")
        recovered = [event]
        next_up = adapter._pending_messages.get("key")
        if next_up is not None:
            recovered.append(next_up)
        recovered.extend(runner._session_state("key").conversation.queued_events)
    assert len(recovered) == 3
    assert recovered[0] is first and recovered[2] is second
    assert recovered[1].text == authored
