---
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
| prompt_template | string | Parameterized prompt (uses `{variable}` syntax) |
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
@software{skillforge2026,
  author = {SkillForge Contributors},
  title = {SkillForge: Trace-to-Skill Distillation Engine},
  year = {2026},
  url = {https://github.com/rohmishra12/skillforge}
}
```

Exported: 2026-10-14
Total skills: 2
