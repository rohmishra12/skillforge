from __future__ import annotations

import pytest
from skillforge.models import Skill, SkillExample, Trace, TraceStep


def make_skill(**kwargs) -> Skill:
    defaults = dict(
        name="test_skill",
        description="does something",
        prompt_template="Do {{task}} in {{style}} style.",
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "style": {"type": "string"},
            },
            "required": ["task", "style"],
        },
    )
    defaults.update(kwargs)
    return Skill(**defaults)


class TestSkillSerialization:
    def test_roundtrip_preserves_all_fields(self):
        skill = make_skill(
            examples=[SkillExample(input={"task": "x", "style": "y"}, expected_output="z")],
            tags=["a", "b"],
            validation_score=0.83,
            version="0.2.1",
        )
        restored = Skill.from_dict(skill.to_dict())

        assert restored.name == skill.name
        assert restored.description == skill.description
        assert restored.prompt_template == skill.prompt_template
        assert restored.validation_score == skill.validation_score
        assert restored.version == skill.version
        assert restored.tags == skill.tags
        assert len(restored.examples) == 1
        assert restored.examples[0].input == {"task": "x", "style": "y"}

    def test_from_dict_handles_missing_optional_fields(self):
        minimal = {"name": "minimal", "description": "", "prompt_template": "", "input_schema": {}}
        skill = Skill.from_dict(minimal)
        assert skill.examples == []
        assert skill.tags == []
        assert skill.validation_score == 0.0

    def test_from_dict_requires_name(self):
        with pytest.raises(KeyError):
            Skill.from_dict({"prompt_template": "x"})


class TestSkillRender:
    def test_substitutes_all_placeholders(self):
        skill = make_skill(prompt_template="Translate {{text}} to {{language}}.")
        result = skill.render({"text": "hello", "language": "French"})
        assert result == "Translate hello to French."

    def test_unknown_placeholder_left_as_is(self):
        skill = make_skill(prompt_template="Hello {{name}}, your {{unknown}} is ready.")
        result = skill.render({"name": "Alice"})
        assert "{{unknown}}" in result
        assert "Alice" in result

    def test_empty_inputs(self):
        skill = make_skill(prompt_template="Static prompt.")
        assert skill.render({}) == "Static prompt."


class TestTrace:
    def test_tool_step_preserves_args(self):
        step = TraceStep(
            role="tool",
            content="called read_file",
            tool_name="read_file",
            tool_args={"path": "/tmp/x.txt"},
            tool_result="file contents",
        )
        trace = Trace(title="test", steps=[step])
        assert trace.steps[0].tool_args == {"path": "/tmp/x.txt"}

    def test_default_id_is_unique(self):
        t1 = Trace(title="a")
        t2 = Trace(title="b")
        assert t1.id != t2.id
