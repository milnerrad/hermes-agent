"""Fail-closed, profile-scoped preparation of human busy-image corrections."""
import base64
import io
import os
from pathlib import Path
import stat

from agent.image_steering import ImageSteer, ImageSubmission

# A busy correction must be cheap and bounded; larger media keeps the normal queue path.
_MAX_BYTES = 20 * 1024 * 1024
_MAX_PIXELS = 40_000_000


def _cached_image(path: str, cache: Path, remaining: int) -> tuple[str, int]:
    """Read only a regular file below this profile's cache, without following symlinks.

    Walk with directory descriptors so a renamed/symlink-swapped ancestor cannot
    redirect the open to an unrelated file after validation. Never fetch URLs.
    """
    from agent.file_safety import raise_if_read_blocked
    from PIL import Image

    relative = Path(path).relative_to(cache)
    if not relative.parts or any(p in ("..", ".") for p in relative.parts):
        raise ValueError("not a cache image")
    raise_if_read_blocked(path)
    if not cache.is_absolute():
        raise ValueError("cache must be absolute")
    fd = os.open(cache.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Protect cache ancestors too; O_NOFOLLOW on the leaf alone is insufficient.
        for component in (*cache.parts[1:], *relative.parts[:-1]):
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        image_fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(image_fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or not 0 < info.st_size <= remaining):
                raise ValueError("invalid image size/type")
            raw = stream.read(remaining + 1)
    finally:
        os.close(fd)
    if len(raw) > remaining:
        raise ValueError("image grew beyond limit")
    with Image.open(io.BytesIO(raw)) as image:
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif", "WEBP": "image/webp"}.get(image.format or "")
        if mime is None or image.width * image.height > _MAX_PIXELS:
            raise ValueError("unsupported image")
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()  # reject truncated/corrupt pixels, not just a plausible header
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", len(raw)


def non_audio_media(event) -> bool:
    """Keep every all-audio event on its original path, not just STT-eligible voices."""
    from gateway.run import _event_media_is_audio
    return bool(event.media_urls) and not all(
        _event_media_is_audio(event, i) for i in range(len(event.media_urls))
    )


async def resolve_busy_images(runner, event, session_key, mode, agent):
    """Order image admissions before any await; decoding speed must not reorder a burst.

    The lock belongs to the existing ephemeral turn-state object (not config or a
    global map). It only serializes media corrections, never reorders the FIFO or
    captures slash commands/Stop, and source-profile contexts remain per task.
    """
    import asyncio
    from gateway.run import _AGENT_PENDING_SENTINEL

    state = runner._peek_session_state(session_key)
    if state is None:
        return runner._BusySteerOutcome("queue", False, False, False, False)
    lock = vars(state.turn).setdefault("_busy_image_lock", asyncio.Lock())
    async with lock:
        subagents = mode in ("interrupt", "steer") and runner._agent_has_active_subagents(agent)
        if subagents:
            mode = runner._demote_interrupt(session_key, "the running agent has active subagents (#30170)")
        compression = mode in ("interrupt", "steer") and await runner._session_has_compression_in_flight(session_key)
        if compression:
            mode = runner._demote_interrupt(session_key, "context compression is in flight (#56391)")
        live = agent is not None and agent is not _AGENT_PENDING_SENTINEL
        accepted = False
        if mode in ("interrupt", "steer") and live and not event.internal:
            accepted = await runner._redirect_busy_images(agent, event, session_key, mode)
        return runner._BusySteerOutcome(
            effective_mode=mode if accepted else "queue",
            demoted_for_subagents=subagents, demoted_for_compression=compression,
            steered=accepted and mode == "steer", redirected=accepted and mode == "interrupt",
        )


def prepare_image_steer(runner, agent, event):
    """None means preserve the original event through normal queue handling."""
    from agent.image_routing import decide_image_input_mode
    from gateway.platforms.base import get_image_cache_dir
    from gateway.run import _event_media_is_image, _load_gateway_config

    if event.internal or event.get_command() or not event.media_urls:
        return None
    if (len(event.media_types) > len(event.media_urls)
            or any(mime and not mime.lower().startswith("image/") for mime in event.media_types)):
        return None
    if not all(_event_media_is_image(event, i) for i in range(len(event.media_urls))):
        return None
    if not callable(getattr(type(agent), "redirect_images", None)):
        return None
    with runner._profile_scope_for_source(event.source):
        cfg = _load_gateway_config()
        if (getattr(agent, "api_mode", None) not in ("chat_completions", "codex_responses")
                or agent._model_supports_vision() is not True
                or decide_image_input_mode(agent.provider, agent.model, cfg) != "native"):
            return None
        text = event.text or "[Images supplied by the user]"
        text = runner._prepend_inbound_reply_context(event, event.source, text)
        text = runner._steer_text_with_origin(text, event)
        cache = get_image_cache_dir()
        urls = []
        remaining = _MAX_BYTES
        try:
            for path in event.media_urls:
                url, size = _cached_image(path, cache, remaining)
                urls.append(url)
                remaining -= size
        except (OSError, ValueError, ImageError):
            return None
    submission = ImageSubmission(text, tuple(urls), event)
    return ImageSteer(text, (submission,))


# Pillow exceptions include decoding errors that are not ValueError/OSError.
from PIL.Image import DecompressionBombError as ImageError
