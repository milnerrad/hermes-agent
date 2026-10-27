"""Internal string-compatible steering envelope; provider rows receive explicit blocks.

Text steers retain their historical ABI. Image-bearing steers carry ordered immutable
submissions through the same locked buffer, including their opaque fallback event.
Never serialize this envelope as text: materialize it at the new-user checkpoint.
"""
import logging
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImageSubmission:
    text: str
    urls: tuple[str, ...]
    fallback: Any


class ImageSteer(str):
    parts: tuple[str | ImageSubmission, ...]
    def __new__(cls, text: str, parts: tuple[str | ImageSubmission, ...]):
        value = super().__new__(cls, text)
        value.parts = parts
        return value

    def strip(self, chars=None):
        return ImageSteer(super().strip(chars), self.parts)

    def content(self, prefix="", suffix="") -> list[dict]:
        blocks = []
        if prefix:
            blocks.append({"type": "text", "text": prefix})
        for part in self.parts:
            text = part.text if isinstance(part, ImageSubmission) else part
            blocks.append({"type": "text", "text": text})
            if isinstance(part, ImageSubmission):
                blocks.extend({"type": "image_url", "image_url": {"url": url}} for url in part.urls)
        if suffix:
            blocks.append({"type": "text", "text": suffix})
        return blocks

    def __getnewargs__(self):
        return str(self), self.parts


def defer_unsupported_images(agent, text) -> bool:
    """A fallback/model switch can remove vision after admission; return original media later."""
    if not isinstance(text, ImageSteer):
        return False
    try:
        supported = (getattr(agent, "api_mode", None) in ("chat_completions", "codex_responses")
                     and agent._model_supports_vision() is True)
    except Exception:
        logging.getLogger(__name__).debug("Image steering capability check failed", exc_info=True)
        supported = False
    if supported:
        return False
    from agent.interrupt_control import requeue_drained_steer
    requeue_drained_steer(agent, text)
    return True


def join_steers(parts, separator="\n"):
    parts = list(parts)
    text = separator.join(parts)
    if not any(isinstance(part, ImageSteer) for part in parts):
        return text
    chunks = []
    for index, part in enumerate(parts):
        if index:
            chunks.append(separator)
        chunks.extend(part.parts if isinstance(part, ImageSteer) else (part,))
    return ImageSteer(text, tuple(chunks))
