from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Config
from .core.observer import ObserverAgent
from .core.synthesizer import SynthesizerAgent
from .core.validator import ValidatorAgent
from .core.optimizer import OptimizerAgent
from .models import Trace, TraceStep, Skill
from .registry.store import SkillRegistry

logger = logging.getLogger(__name__)


class SkillForgeHarness:
    """
    Orchestrates the four-stage synthesis pipeline:
    Trace → observe → synthesize → (validate → optimize)* → Skill

    The validate/optimize loop runs up to config.max_optimization_rounds times,
    stopping early when the skill passes (score ≥ 0.70).
    """

    def __init__(self, config: Config, registry: SkillRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or SkillRegistry()
        self._observer = ObserverAgent(config)
        self._synthesizer = SynthesizerAgent(config)
        self._validator = ValidatorAgent(config)
        self._optimizer = OptimizerAgent(config)

    def run(self, trace: Trace, *, save: bool = True) -> Skill:
        """
        Run the full pipeline on a trace.

        Raises LLMError if any stage fails. The caller is responsible for
        deciding whether to retry or surface the error to the user.
        """
        observation = self._observer.observe(trace)
        skill = self._synthesizer.synthesize(observation)

        for round_num in range(1, self.config.max_optimization_rounds + 1):
            report = self._validator.validate(skill)
            skill.validation_score = report.score

            if report.passed:
                logger.info(
                    "skill '%s' passed validation on round %d (score=%.2f)",
                    skill.name, round_num, report.score,
                )
                break

            logger.info(
                "skill '%s' round %d score=%.2f — optimizing (%d failure mode(s))",
                skill.name, round_num, report.score, len(report.failure_modes),
            )
            if round_num < self.config.max_optimization_rounds:
                skill = self._optimizer.optimize(skill, report)
        else:
            logger.warning(
                "skill '%s' did not pass after %d round(s) (final score=%.2f)",
                skill.name,
                self.config.max_optimization_rounds,
                skill.validation_score,
            )

        if save:
            self.registry.save(skill)

        return skill

    @staticmethod
    def load_trace(path: Path | str) -> Trace:
        """
        Load a Trace from a JSON file. See examples/traces/ for the format.

        Security note: trace content is passed verbatim to LLM prompts. Only load
        traces from sources you trust. Malicious trace content could attempt prompt
        injection against the synthesis agents.
        """
        path = Path(path)

        # Guard against very large files before reading into memory.
        max_bytes = 512 * 1024  # 512 KB
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"Trace file is {size // 1024} KB — exceeds the {max_bytes // 1024} KB limit. "
                "Split the trace or increase the limit explicitly."
            )

        data = json.loads(path.read_text(encoding="utf-8"))

        # Restrict step roles to known values to catch malformed traces early.
        _valid_roles = {"user", "assistant", "tool", "system"}
        steps = []
        for i, s in enumerate(data.get("steps", [])):
            role = s.get("role", "")
            if role not in _valid_roles:
                raise ValueError(
                    f"Step {i} has unknown role '{role}'. Expected one of: {sorted(_valid_roles)}"
                )
            # Coerce content to string — defensively handle non-string values.
            steps.append(
                TraceStep(
                    role=role,
                    content=str(s.get("content", "")),
                    tool_name=s.get("tool_name"),
                    tool_args=s.get("tool_args") if isinstance(s.get("tool_args"), dict) else None,
                    tool_result=str(s["tool_result"]) if s.get("tool_result") is not None else None,
                    timestamp=s.get("timestamp", ""),
                )
            )

        return Trace(
            id=data.get("id", ""),
            title=str(data.get("title", path.stem))[:256],  # cap title length
            steps=steps,
            metadata=data.get("metadata", {}),
        )
