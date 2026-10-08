from __future__ import annotations

import copy
import json
import logging
import textwrap
from datetime import datetime, timezone

from ..config import Config
from ..llm import complete_json
from ..models import Skill, SkillExample
from .validator import ValidationReport

logger = logging.getLogger(__name__)

_SYSTEM = textwrap.dedent("""\
    You improve LLM prompt templates based on observed failure modes.
    Make targeted fixes — preserve what works. If the prompt is ambiguous,
    make it concrete. If it over-constrains, loosen it.
    The prompt_template must remain an instruction prompt for an LLM with
    simple {{variable}} placeholders (no loops or handlebars tags).
    Output only JSON.
""")

_USER = textwrap.dedent("""\
    Improve this skill. Its validation score is {score:.2f} (target ≥ 0.70).

    Current prompt template:
    {template}

    Failure modes:
    {failures}

    Failed examples (up to 3):
    {failed_examples}

    Return JSON:
    {{
      "prompt_template": "improved instruction prompt for an LLM with {{{{variable}}}} placeholders",
      "description": "updated description if needed, else repeat the original",
      "new_examples": [
        {{
          "input": {{}},
          "expected_output": "description",
          "description": "what case this covers"
        }}
      ],
      "changes": ["concise description of each change made"]
    }}

    Rules:
    - prompt_template must be an instruction prompt directing an LLM to perform the task
    - use only simple {{{{variable}}}} placeholders matching input variables
    - do not use handlebars or template tags (such as {{{{this}}}}, {{{{/each}}}}, loops)
""")


class OptimizerAgent:
    """Rewrites a skill's prompt template to address validation failures."""

    def __init__(self, config: Config) -> None:
        self.config = config

    @staticmethod
    def _bump_version(v: str) -> str:
        parts = v.split(".")
        try:
            parts[-1] = str(int(parts[-1]) + 1)
        except (ValueError, IndexError):
            parts.append("1")
        return ".".join(parts)

    def optimize(self, skill: Skill, report: ValidationReport) -> Skill:
        """Return an improved copy of skill. Returns skill unchanged if score ≥ 0.90."""
        if report.score >= 0.90:
            return skill

        failed = [r for r in report.example_results if r["score"] < 0.7]

        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER.format(
                    score=report.score,
                    template=skill.prompt_template,
                    failures=json.dumps(report.failure_modes, indent=2),
                    failed_examples=json.dumps(failed[:3], indent=2),
                ),
            },
        ]
        raw = complete_json(messages, self.config)

        improved = copy.deepcopy(skill)
        improved.prompt_template = raw.get("prompt_template", skill.prompt_template)
        improved.description = raw.get("description", skill.description)
        improved.version = self._bump_version(skill.version)
        improved.updated_at = datetime.now(timezone.utc).isoformat()

        for ex_data in raw.get("new_examples", []):
            improved.examples.append(
                SkillExample(
                    input=ex_data["input"],
                    expected_output=ex_data["expected_output"],
                    description=ex_data.get("description", ""),
                )
            )

        changes = raw.get("changes", [])
        logger.info(
            "optimized '%s' → v%s (%d change(s))",
            skill.name,
            improved.version,
            len(changes),
        )
        return improved
