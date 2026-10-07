from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI, APIError

from .config import Config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails in a way we can't recover from."""


def _client(config: Config) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def complete(
    messages: list[dict[str, str]],
    config: Config,
    *,
    json_mode: bool = False,
) -> str:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = _client(config).chat.completions.create(**kwargs)
    except APIError as exc:
        raise LLMError(f"API call failed: {exc}") from exc

    content = response.choices[0].message.content or ""
    logger.debug("LLM response (%d chars)", len(content))
    return content


def complete_json(messages: list[dict[str, str]], config: Config) -> dict[str, Any]:
    raw = complete(messages, config, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model returned invalid JSON: {exc}\n\nRaw output:\n{raw}") from exc
