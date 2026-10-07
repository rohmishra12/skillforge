from __future__ import annotations

import json
import pytest
from pathlib import Path

from skillforge.harness import SkillForgeHarness


@pytest.fixture
def trace_file(tmp_path) -> Path:
    data = {
        "id": "t-001",
        "title": "Weekly email draft",
        "steps": [
            {"role": "user", "content": "Turn these bullets into an email."},
            {"role": "assistant", "content": "Here's a draft..."},
            {
                "role": "tool",
                "content": "fetch doc",
                "tool_name": "read_file",
                "tool_args": {"path": "/tmp/bullets.txt"},
                "tool_result": "- shipped auth\n- fixed 3 bugs",
            },
        ],
    }
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(data))
    return p


def test_load_trace_parses_all_fields(trace_file):
    trace = SkillForgeHarness.load_trace(trace_file)
    assert trace.id == "t-001"
    assert trace.title == "Weekly email draft"
    assert len(trace.steps) == 3
    assert trace.steps[2].tool_name == "read_file"
    assert trace.steps[2].tool_args == {"path": "/tmp/bullets.txt"}


def test_load_trace_falls_back_to_filename_as_title(tmp_path):
    p = tmp_path / "my_workflow.json"
    p.write_text(json.dumps({"steps": [{"role": "user", "content": "hi"}]}))
    trace = SkillForgeHarness.load_trace(p)
    assert trace.title == "my_workflow"


def test_load_trace_empty_steps(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"title": "Empty", "steps": []}))
    trace = SkillForgeHarness.load_trace(p)
    assert trace.steps == []


def test_load_trace_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        SkillForgeHarness.load_trace(Path("/nonexistent/trace.json"))
