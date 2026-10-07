"""
Configuration for the SkillForge pipeline.

All settings are read from environment variables at import time and exposed
as a single Config instance. Agents receive a Config; they don't read env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    base_url: str | None
    max_tokens: int
    temperature: float
    max_optimization_rounds: int

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.environ.get("SKILLFORGE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "No API key found. Set SKILLFORGE_API_KEY or OPENAI_API_KEY."
            )
        return cls(
            api_key=api_key,
            model=os.environ.get("SKILLFORGE_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("SKILLFORGE_BASE_URL") or None,
            max_tokens=int(os.environ.get("SKILLFORGE_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get("SKILLFORGE_TEMPERATURE", "0.2")),
            max_optimization_rounds=int(os.environ.get("SKILLFORGE_ROUNDS", "2")),
        )


# Module-level default — agents can use this or accept an explicit Config.
default_config = Config.from_env  # callable, not eagerly evaluated
