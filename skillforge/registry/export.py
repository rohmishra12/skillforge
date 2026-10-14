"""
Export skills to HuggingFace-compatible dataset format.
Produces a JSONL file per the HF datasets convention, plus a dataset card (README.md).
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from ..models import Skill


DATASET_CARD_TEMPLATE = """---
language:
- en
license: apache-2.0
tags:
- skillforge
- agentic-ai
- prompt-engineering
- skills
task_categories:
- text-generation
pretty_name: SkillForge Skill Dataset
---

# SkillForge Skill Dataset

This dataset was exported from [SkillForge](https://github.com/rohmishra12/skillforge),
a trace-to-skill distillation engine.

## Dataset Description

Each record is a verified, reusable skill synthesized from execution traces.

## Data Fields

| Field | Type | Description |
|---|---|---|
| id | string | Unique skill ID |
| name | string | Snake-case skill name |
| description | string | One-sentence description |
| prompt_template | string | Parameterized prompt (uses `{{variable}}` syntax) |
| input_schema | object | JSON Schema for template variables |
| examples | list | Input/output example pairs |
| tags | list | Topic tags |
| version | string | Skill version |
| validation_score | float | 0.0-1.0 quality score from ValidatorAgent |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("rohmishra12/skillforge-skills")
for skill in ds["train"]:
    print(skill["name"], skill["validation_score"])
```

## Citation

```bibtex
@software{{skillforge2026,
  author = {{SkillForge Contributors}},
  title = {{SkillForge: Trace-to-Skill Distillation Engine}},
  year = {{2026}},
  url = {{https://github.com/rohmishra12/skillforge}}
}}
```

Exported: {export_date}
Total skills: {total_skills}
"""


def export_to_huggingface(
    skills: list[Skill],
    output_dir: Path | str,
    min_score: float = 0.5,
) -> Path:
    """
    Export a list of skills to a HuggingFace-compatible dataset directory.

    Args:
        skills: List of Skill objects to export
        output_dir: Directory to write the dataset to
        min_score: Only export skills with validation_score >= this value

    Returns:
        Path to the output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qualified = [s for s in skills if s.validation_score >= min_score]

    # Write JSONL
    jsonl_path = output_dir / "data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for skill in qualified:
            record = skill.to_dict()
            # Flatten examples to JSON strings for HF compatibility
            record["examples"] = json.dumps(record["examples"])
            record["input_schema"] = json.dumps(record["input_schema"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write dataset card
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        DATASET_CARD_TEMPLATE.format(
            export_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            total_skills=len(qualified),
        ),
        encoding="utf-8",
    )

    return output_dir
