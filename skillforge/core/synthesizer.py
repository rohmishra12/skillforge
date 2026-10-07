from __future__ import annotations

import logging
import textwrap
import uuid
from datetime import datetime, timezone
from typing import Any

from ..config import Config
from ..llm import complete_json
from ..models import Skill, SkillExample

logger = logging.getLogger(__name__)

_SYSTEM = textwrap.dedent("""\
    You write production-quality, parameterized LLM prompt templates.
    A good template is specific enough to be useful and general enough
    to work across the full range of valid inputs. Use {{variable}} for
    interpolated values. Respond only with the JSON schema requested.
""")

_USER = textwrap.dedent("""\
    Synthesize a reusable skill from this workflow observation.

    {observation}

    Return JSON:
    {{
      "name": "snake_case_name",
      "description": "one sentence — what it does and when to use it",
      "prompt_template": "complete prompt with {{{{variable}}}} placeholders",
      "input_schema": {{
        "type": "object",
        "properties": {{
          "var": {{"type": "string", "description": "..."}}
        }},
        "required": ["var"]
      }},
      "examples": [
        {{
          "input": {{"var": "value"}},
          "expected_output": "description of expected result",
          "description": "label"
        }}
      ],
      "tags": ["tag"]
    }}

    Rules:
    - prompt_template must be complete and self-contained
    - provide at least 2 examples that exercise different input ranges
    - do not invent functionality not present in the observation
""")


class SynthesizerAgent:
    """Turns an observation dict into a validated Skill object."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def synthesize(self, observation: dict[str, Any]) -> Skill:
        import json

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER.format(observation=json.dumps(observation, indent=2))},
        ]
        raw = complete_json(messages, self.config)

        skill = Skill(
            id=str(uuid.uuid4()),
            name=raw["name"],
            description=raw.get("description", ""),
            prompt_template=raw.get("prompt_template", ""),
            input_schema=raw.get("input_schema", {}),
            examples=[
                SkillExample(
                    input=e["input"],
                    expected_output=e["expected_output"],
                    description=e.get("description", ""),
                )
                for e in raw.get("examples", [])
            ],
            tags=raw.get("tags", []),
            source_trace_id=observation.get("source_trace_id"),
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("synthesized skill '%s'", skill.name)
        return skill
