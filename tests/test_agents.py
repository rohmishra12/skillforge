"""
Agent tests using recorded fixtures — no live API calls required.

Each test patches skillforge.llm.complete / complete_json with fixture responses,
then runs the real agent code against those canned responses.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from skillforge.config import Config
from skillforge.models import Skill, SkillExample, Trace, TraceStep
from skillforge.core.observer import ObserverAgent
from skillforge.core.synthesizer import SynthesizerAgent
from skillforge.core.validator import ValidatorAgent, PASS_THRESHOLD
from skillforge.core.optimizer import OptimizerAgent

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def config() -> Config:
    """Minimal config — API key is never used because LLM calls are mocked."""
    return Config(
        api_key="test-key",
        model="gpt-4o-mini",
        base_url=None,
        max_tokens=2048,
        temperature=0.2,
        max_optimization_rounds=2,
    )


@pytest.fixture
def weekly_email_skill() -> Skill:
    fx = _load("synthesizer_weekly_email.json")
    r = fx["response"]
    return Skill(
        name=r["name"],
        description=r["description"],
        prompt_template=r["prompt_template"],
        input_schema=r["input_schema"],
        examples=[
            SkillExample(
                input=e["input"],
                expected_output=e["expected_output"],
                description=e.get("description", ""),
            )
            for e in r["examples"]
        ],
        tags=r.get("tags", []),
        source_trace_id="trace-001",
    )


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class TestObserverAgent:
    def test_observe_returns_goal_and_variables(self, config):
        fx = _load("observer_weekly_email.json")
        trace = Trace(
            title="Draft a weekly status email from bullet points",
            steps=[
                TraceStep(role="user", content="Turn these bullets into an email."),
                TraceStep(role="assistant", content="Here's a draft..."),
            ],
        )

        with patch("skillforge.core.observer.complete_json", return_value=fx["response"]):
            result = ObserverAgent(config).observe(trace)

        assert result["goal"] == fx["response"]["goal"]
        assert len(result["input_variables"]) == len(fx["response"]["input_variables"])
        assert "source_trace_id" in result
        assert result["source_trace_id"] == trace.id

    def test_observe_preserves_source_trace_id(self, config):
        fx = _load("observer_weekly_email.json")
        trace = Trace(title="test", steps=[TraceStep(role="user", content="hi")])

        with patch("skillforge.core.observer.complete_json", return_value=fx["response"]):
            result = ObserverAgent(config).observe(trace)

        assert result["source_trace_id"] == trace.id

    def test_observe_includes_tags(self, config):
        fx = _load("observer_weekly_email.json")
        trace = Trace(title="test", steps=[TraceStep(role="user", content="hi")])

        with patch("skillforge.core.observer.complete_json", return_value=fx["response"]):
            result = ObserverAgent(config).observe(trace)

        assert "tags" in result
        assert isinstance(result["tags"], list)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

class TestSynthesizerAgent:
    def test_synthesize_returns_skill(self, config):
        fx = _load("synthesizer_weekly_email.json")

        with patch("skillforge.core.synthesizer.complete_json", return_value=fx["response"]):
            skill = SynthesizerAgent(config).synthesize(fx["request"]["observation"])

        assert skill.name == "draft_weekly_status_email"
        assert skill.prompt_template != ""
        assert "{{manager_name}}" in skill.prompt_template

    def test_synthesize_populates_examples(self, config):
        fx = _load("synthesizer_weekly_email.json")

        with patch("skillforge.core.synthesizer.complete_json", return_value=fx["response"]):
            skill = SynthesizerAgent(config).synthesize(fx["request"]["observation"])

        assert len(skill.examples) == 2
        assert skill.examples[0].input["manager_name"] == "John Doe"

    def test_synthesize_sets_source_trace_id(self, config):
        fx = _load("synthesizer_weekly_email.json")
        obs = dict(fx["request"]["observation"])
        obs["source_trace_id"] = "trace-001"

        with patch("skillforge.core.synthesizer.complete_json", return_value=fx["response"]):
            skill = SynthesizerAgent(config).synthesize(obs)

        assert skill.source_trace_id == "trace-001"

    def test_synthesize_input_schema_has_required_fields(self, config):
        fx = _load("synthesizer_weekly_email.json")

        with patch("skillforge.core.synthesizer.complete_json", return_value=fx["response"]):
            skill = SynthesizerAgent(config).synthesize(fx["request"]["observation"])

        required = skill.input_schema.get("required", [])
        assert "bullet_points" in required
        assert "manager_name" in required


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class TestValidatorAgent:
    def test_validate_passing_skill_returns_score_above_threshold(self, config, weekly_email_skill):
        fx = _load("validator_weekly_email.json")
        exec_responses = iter(fx["execution_responses"])
        judge_responses = iter(fx["judge_responses"])

        def mock_complete(messages, cfg, **kwargs):
            return next(exec_responses)

        def mock_complete_json(messages, cfg):
            # adversarial input generation also uses complete_json — return empty cases
            msg_text = " ".join(m.get("content", "") for m in messages)
            if "adversarial" in msg_text.lower() or "edge-case" in msg_text.lower():
                return {"cases": []}
            return next(judge_responses)

        with patch("skillforge.core.validator.complete", side_effect=mock_complete), \
             patch("skillforge.core.validator.complete_json", side_effect=mock_complete_json):
            report = ValidatorAgent(config).validate(weekly_email_skill)

        assert report.score >= PASS_THRESHOLD
        assert report.passed is True

    def test_validate_no_examples_returns_zero(self, config):
        empty_skill = Skill(
            name="empty",
            description="no examples",
            prompt_template="Do {{task}}.",
            input_schema={"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
        )
        report = ValidatorAgent(config).validate(empty_skill)
        assert report.score == 0.0
        assert report.passed is False
        assert "no examples provided" in report.failure_modes

    def test_validate_report_has_all_fields(self, config, weekly_email_skill):
        fx = _load("validator_weekly_email.json")
        exec_responses = iter(fx["execution_responses"])
        judge_responses = iter(fx["judge_responses"])

        def mock_complete(messages, cfg, **kwargs):
            return next(exec_responses)

        def mock_complete_json(messages, cfg):
            msg_text = " ".join(m.get("content", "") for m in messages)
            if "adversarial" in msg_text.lower() or "edge-case" in msg_text.lower():
                return {"cases": []}
            return next(judge_responses)

        with patch("skillforge.core.validator.complete", side_effect=mock_complete), \
             patch("skillforge.core.validator.complete_json", side_effect=mock_complete_json):
            report = ValidatorAgent(config).validate(weekly_email_skill)

        assert isinstance(report.score, float)
        assert isinstance(report.passed, bool)
        assert isinstance(report.example_results, list)
        assert isinstance(report.failure_modes, list)
        assert len(report.example_results) == len(weekly_email_skill.examples)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class TestOptimizerAgent:
    def test_optimize_returns_new_skill_with_bumped_version(self, config, weekly_email_skill):
        fx = _load("optimizer_weekly_email.json")

        from skillforge.core.validator import ValidationReport
        failing_report = ValidationReport(
            score=fx["request"]["score"],
            example_results=[
                {
                    "input": {"bullet_points": "task", "week_date": "Oct 1", "manager_name": "Bob", "your_name": "Alice"},
                    "expected": "formal email",
                    "actual": "hey bob, did stuff this week",
                    "score": 0.1,
                    "reasoning": "Not formal",
                    "issues": fx["request"]["failure_modes"],
                }
            ],
            adversarial_results=[],
            failure_modes=fx["request"]["failure_modes"],
        )

        with patch("skillforge.core.optimizer.complete_json", return_value=fx["response"]):
            improved = OptimizerAgent(config).optimize(weekly_email_skill, failing_report)

        assert improved.version != weekly_email_skill.version
        assert improved.prompt_template != weekly_email_skill.prompt_template
        assert "{{manager_name}}" in improved.prompt_template

    def test_optimize_adds_new_examples(self, config, weekly_email_skill):
        fx = _load("optimizer_weekly_email.json")

        from skillforge.core.validator import ValidationReport
        failing_report = ValidationReport(
            score=0.10,
            example_results=[{
                "input": {}, "expected": "", "actual": "",
                "score": 0.1, "reasoning": "", "issues": ["Not formal"]
            }],
            adversarial_results=[],
            failure_modes=["Not formal"],
        )

        original_example_count = len(weekly_email_skill.examples)

        with patch("skillforge.core.optimizer.complete_json", return_value=fx["response"]):
            improved = OptimizerAgent(config).optimize(weekly_email_skill, failing_report)

        assert len(improved.examples) > original_example_count

    def test_optimize_skips_if_score_already_high(self, config, weekly_email_skill):
        from skillforge.core.validator import ValidationReport
        passing_report = ValidationReport(
            score=0.95,
            example_results=[],
            adversarial_results=[],
            failure_modes=[],
        )

        # No LLM call should happen
        with patch("skillforge.core.optimizer.complete_json") as mock_llm:
            result = OptimizerAgent(config).optimize(weekly_email_skill, passing_report)
            mock_llm.assert_not_called()

        assert result is weekly_email_skill  # same object returned unchanged
