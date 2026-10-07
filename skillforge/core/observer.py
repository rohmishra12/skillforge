from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from ..config import Config
from ..llm import complete_json
from ..models import Trace, TraceStep

logger = logging.getLogger(__name__)

_SYSTEM = textwrap.dedent("""\
    You extract the reusable, parameterized intent from a noisy execution trace.
    Strip out errors, retries, and one-off details. Focus on what a practitioner
    would need to reproduce the outcome on different inputs.
    Respond only with the JSON schema requested — no commentary.
""")

_USER = textwrap.dedent("""\
    Trace title: {title}

    Steps:
    {steps}

    Return JSON:
    {{
      "goal": "one sentence",
      "input_variables": [
        {{"name": "...", "description": "...", "example": "..."}}
      ],
      "key_steps": [
        {{"step": 1, "action": "...", "why": "..."}}
      ],
      "output_description": "what the workflow produces",
      "constraints": ["..."],
      "tags": ["..."]
    }}
""")


def _format_steps(steps: list[TraceStep]) -> str:
    lines = []
    for i, s in enumerate(steps, 1):
        line = f"[{i}] {s.role.upper()}: {s.content}"
        if s.tool_name:
            args = json.dumps(s.tool_args or {})
            line += f"\n     call: {s.tool_name}({args})"
        if s.tool_result:
            preview = s.tool_result[:200] + "…" if len(s.tool_result) > 200 else s.tool_result
            line += f"\n     result: {preview}"
        lines.append(line)
    return "\n".join(lines)


class ObserverAgent:
    """Extracts structured intent from a raw execution trace."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def observe(self, trace: Trace) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER.format(
                    title=trace.title,
                    steps=_format_steps(trace.steps),
                ),
            },
        ]
        result = complete_json(messages, self.config)
        result["source_trace_id"] = trace.id
        logger.info("observed trace %s: %s", trace.id[:8], result.get("goal", ""))
        return result
