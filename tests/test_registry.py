from __future__ import annotations

import pytest
from skillforge.models import Skill, SkillExample
from skillforge.registry.store import SkillRegistry


@pytest.fixture
def registry(tmp_path):
    return SkillRegistry(db_path=tmp_path / "test.db")


def skill(name: str, score: float = 0.8) -> Skill:
    return Skill(
        name=name,
        description=f"{name} skill",
        prompt_template="Do {{task}}.",
        input_schema={"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
        examples=[SkillExample(input={"task": "something"}, expected_output="result")],
        validation_score=score,
    )


class TestRegistryCRUD:
    def test_save_and_get(self, registry):
        s = skill("email_drafter")
        registry.save(s)
        fetched = registry.get(s.id)
        assert fetched is not None
        assert fetched.name == "email_drafter"
        assert fetched.validation_score == 0.8

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("does-not-exist") is None

    def test_save_upserts_on_same_id(self, registry):
        s = skill("updatable")
        registry.save(s)
        s.description = "updated"
        registry.save(s)
        assert registry.count() == 1
        assert registry.get(s.id).description == "updated"

    def test_delete_existing(self, registry):
        s = skill("to_delete")
        registry.save(s)
        assert registry.delete(s.id) is True
        assert registry.get(s.id) is None

    def test_delete_nonexistent_returns_false(self, registry):
        assert registry.delete("ghost") is False


class TestRegistryQuery:
    def test_list_all_ordered_by_updated_at(self, registry):
        registry.save(skill("a"))
        registry.save(skill("b"))
        registry.save(skill("c"))
        names = [s.name for s in registry.list_all()]
        assert len(names) == 3

    def test_list_all_min_score_filter(self, registry):
        registry.save(skill("high", score=0.9))
        registry.save(skill("mid",  score=0.6))
        registry.save(skill("low",  score=0.3))
        results = registry.list_all(min_score=0.7)
        assert len(results) == 1
        assert results[0].name == "high"

    def test_search_by_name_substring(self, registry):
        registry.save(skill("email_drafter"))
        registry.save(skill("code_reviewer"))
        results = registry.search("email")
        assert len(results) == 1
        assert results[0].name == "email_drafter"

    def test_search_by_description(self, registry):
        s = skill("some_skill")
        s.description = "helps with quarterly reports"
        registry.save(s)
        assert len(registry.search("quarterly")) == 1
        assert len(registry.search("nonexistent")) == 0

    def test_count(self, registry):
        assert registry.count() == 0
        registry.save(skill("x"))
        registry.save(skill("y"))
        assert registry.count() == 2
