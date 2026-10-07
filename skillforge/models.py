from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


@dataclass
class TraceStep:
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    timestamp: str = field(default_factory=_now)


@dataclass
class Trace:
    title: str
    steps: list[TraceStep] = field(default_factory=list)
    id: str = field(default_factory=_uid)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)


@dataclass
class SkillExample:
    input: dict[str, Any]
    expected_output: str
    description: str = ""


@dataclass
class Skill:
    """
    A verified, parameterized skill synthesized from an execution trace.

    prompt_template uses {{variable}} placeholders that map to keys in input_schema.
    validation_score is set by the validator; 0.0 until first validation run.
    """

    name: str
    description: str
    prompt_template: str
    input_schema: dict[str, Any]
    examples: list[SkillExample] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=_uid)
    source_trace_id: str | None = None
    version: str = "0.1.0"
    validation_score: float = 0.0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def render(self, inputs: dict[str, Any]) -> str:
        """Substitute {{key}} placeholders with values from inputs."""
        result = self.prompt_template
        for key, value in inputs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "prompt_template": self.prompt_template,
            "input_schema": self.input_schema,
            "examples": [
                {
                    "input": e.input,
                    "expected_output": e.expected_output,
                    "description": e.description,
                }
                for e in self.examples
            ],
            "tags": self.tags,
            "source_trace_id": self.source_trace_id,
            "version": self.version,
            "validation_score": self.validation_score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Skill":
        return cls(
            id=d.get("id", _uid()),
            name=d["name"],
            description=d.get("description", ""),
            prompt_template=d.get("prompt_template", ""),
            input_schema=d.get("input_schema", {}),
            examples=[
                SkillExample(
                    input=e["input"],
                    expected_output=e["expected_output"],
                    description=e.get("description", ""),
                )
                for e in d.get("examples", [])
            ],
            tags=d.get("tags", []),
            source_trace_id=d.get("source_trace_id"),
            version=d.get("version", "0.1.0"),
            validation_score=d.get("validation_score", 0.0),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
        )
