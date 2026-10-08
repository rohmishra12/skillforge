from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from ..config import Config
from ..llm import complete, complete_json, LLMError
from ..models import Skill

logger = logging.getLogger(__name__)

# Threshold below which a skill is considered failing.
PASS_THRESHOLD = 0.7

_JUDGE_SYSTEM = textwrap.dedent("""\
    You evaluate whether an LLM skill produced an acceptable result.
    Be objective. Partial credit is appropriate when the output is directionally
    correct but missing detail or has minor issues.
    Respond only with the JSON schema requested.
""")

_JUDGE_USER = textwrap.dedent("""\
    Skill: {name}
    Description: {description}

    Input:
    {input}

    Expected (description of acceptable output):
    {expected}

    Actual output:
    {actual}

    Return JSON:
    {{
      "score": 0.0,
      "reasoning": "...",
      "issues": ["..."]
    }}

    score is 0.0–1.0. issues is an empty list if score >= 0.9.
""")

_ADVERSARIAL_USER = textwrap.dedent("""\
    Generate 3 adversarial or edge-case inputs for this skill that might expose
    weaknesses: empty values, very long inputs, unexpected types, or boundary cases.

    Skill: {name}
    Description: {description}
    Input schema: {schema}

    Return JSON:
    {{
      "cases": [
        {{"input": {{}}, "rationale": "why this is tricky"}}
      ]
    }}
""")


class ValidationReport:
    __slots__ = ("score", "passed", "example_results", "adversarial_results", "failure_modes")

    def __init__(
        self,
        score: float,
        example_results: list[dict],
        adversarial_results: list[dict],
        failure_modes: list[str],
    ) -> None:
        self.score = score
        self.passed = score >= PASS_THRESHOLD
        self.example_results = example_results
        self.adversarial_results = adversarial_results
        self.failure_modes = failure_modes

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "example_results": self.example_results,
            "adversarial_results": self.adversarial_results,
            "failure_modes": self.failure_modes,
        }


class ValidatorAgent:
    """
    Tests a skill against its declared examples and adversarially-generated inputs.

    Uses an LLM judge to score outputs — this avoids hardcoding expected strings
    while still catching meaningful regressions.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def _run(self, skill: Skill, inputs: dict[str, Any]) -> str:
        prompt = skill.render(inputs)
        return complete([{"role": "user", "content": prompt}], self.config)

    def _judge(self, skill: Skill, inputs: dict, expected: str, actual: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": _JUDGE_USER.format(
                    name=skill.name,
                    description=skill.description,
                    input=json.dumps(inputs, indent=2),
                    expected=expected,
                    actual=actual[:3000],
                ),
            },
        ]
        return complete_json(messages, self.config)

    def _adversarial_inputs(self, skill: Skill) -> list[dict[str, Any]]:
        messages = [
            {
                "role": "user",
                "content": _ADVERSARIAL_USER.format(
                    name=skill.name,
                    description=skill.description,
                    schema=json.dumps(skill.input_schema, indent=2),
                ),
            }
        ]
        try:
            result = complete_json(messages, self.config)
            return result.get("cases", [])
        except LLMError:
            logger.warning("adversarial input generation failed for '%s'", skill.name)
            return []

    def validate(self, skill: Skill) -> ValidationReport:
        if not skill.examples:
            return ValidationReport(
                score=0.0,
                example_results=[],
                adversarial_results=[],
                failure_modes=["no examples provided"],
            )

        example_results = []
        scores = []

        for ex in skill.examples:
            try:
                actual = self._run(skill, ex.input)
                judgement = self._judge(skill, ex.input, ex.expected_output, actual)
                score = float(judgement.get("score", 0.0))
            except LLMError as exc:
                logger.warning("execution failed for example in '%s': %s", skill.name, exc)
                score = 0.0
                judgement = {"score": 0.0, "reasoning": str(exc), "issues": [str(exc)]}
                actual = ""

            scores.append(score)
            example_results.append(
                {
                    "input": ex.input,
                    "expected": ex.expected_output,
                    "actual": actual[:500],
                    "score": score,
                    "reasoning": judgement.get("reasoning", ""),
                    "issues": judgement.get("issues", []),
                }
            )

        adversarial_results = []
        for case in self._adversarial_inputs(skill):
            try:
                actual = self._run(skill, case.get("input", {}))
                adversarial_results.append(
                    {
                        "input": case.get("input", {}),
                        "rationale": case.get("rationale", ""),
                        "actual": actual[:300],
                        "errored": False,
                    }
                )
            except Exception as exc:
                adversarial_results.append(
                    {
                        "input": case.get("input", {}),
                        "rationale": case.get("rationale", ""),
                        "errored": True,
                        "error": str(exc),
                    }
                )

        failure_modes = list(
            {issue for r in example_results for issue in r.get("issues", [])}
        )
        avg = sum(scores) / len(scores) if scores else 0.0

        logger.info("validated '%s': score=%.2f passed=%s", skill.name, avg, avg >= PASS_THRESHOLD)
        return ValidationReport(
            score=round(avg, 3),
            example_results=example_results,
            adversarial_results=adversarial_results,
            failure_modes=failure_modes,
        )
