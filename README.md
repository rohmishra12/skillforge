# SkillForge

SkillForge distills execution traces into verified, parameterized prompt skills. Give it a
conversation log or tool-call transcript; it returns a structured skill artifact with a tested
prompt template, input schema, and usage examples — ready to invoke, version, and share.

## Motivation

Prompt engineering for recurring workflows is tedious and the results are usually stored as
ad-hoc snippets. SkillForge treats skills as first-class artifacts: synthesized by a pipeline
of specialized agents, validated before being saved, and stored in a local registry that can
be exported as a HuggingFace dataset.

## How it works

Four agents run in sequence on each trace:

1. **Observer** — strips noise from the trace and extracts the goal, input variables, and key steps.
2. **Synthesizer** — writes a production-quality parameterized prompt template and example set.
3. **Validator** — executes the skill against its examples and scores each output with an LLM judge.
4. **Optimizer** — rewrites the prompt to fix identified failure modes; repeats until the skill passes or the round limit is hit.

The `SkillForgeHarness` coordinates the loop. Skills that don't reach the passing threshold
(0.70) are still saved — the validation score is visible in the registry so you can decide
whether to manually refine or discard them.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/rohmishra12/skillforge
cd skillforge
pip install -e ".[dev]"
```

```bash
# OpenAI (default)
export SKILLFORGE_API_KEY=sk-...
export SKILLFORGE_MODEL=gpt-4o-mini

# Any OpenAI-compatible endpoint (Groq, Together, Ollama, etc.)
export SKILLFORGE_BASE_URL=http://localhost:11434/v1
export SKILLFORGE_MODEL=llama3.2
```

## Usage

### CLI

```bash
# Synthesize a skill from a trace file
skillforge generate examples/traces/weekly_status_email.json

# List all skills in the local registry
skillforge list

# Inspect a skill (accepts full id or unambiguous prefix)
skillforge show a3f9c1b2

# Re-run validation on an existing skill
skillforge test a3f9c1b2

# Export to a HuggingFace-compatible dataset directory
skillforge export ./dataset/ --min-score 0.7
```

### Python API

```python
from skillforge.config import Config
from skillforge.harness import SkillForgeHarness
from skillforge.models import Trace, TraceStep

trace = Trace(
    title="Summarize a support ticket",
    steps=[
        TraceStep(role="user", content="Summarize this ticket: ..."),
        TraceStep(role="assistant", content="The customer is reporting ..."),
    ],
)

config = Config.from_env()
harness = SkillForgeHarness(config)
skill = harness.run(trace)

print(skill.name)              # e.g. "summarize_support_ticket"
print(skill.validation_score)  # e.g. 0.84
print(skill.render({"ticket": "Login fails on iOS 17"}))
```

### Trace format

```json
{
  "title": "My workflow",
  "steps": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." },
    {
      "role":        "tool",
      "content":     "...",
      "tool_name":   "read_file",
      "tool_args":   { "path": "/tmp/doc.txt" },
      "tool_result": "..."
    }
  ]
}
```

See `examples/traces/` for complete examples.

## Configuration

All settings are read from environment variables. None have required defaults except the API key.

| Variable | Default | Description |
|---|---|---|
| `SKILLFORGE_API_KEY` | — | API key (falls back to `OPENAI_API_KEY`) |
| `SKILLFORGE_MODEL` | `gpt-4o-mini` | Model name |
| `SKILLFORGE_BASE_URL` | OpenAI default | Override for compatible endpoints |
| `SKILLFORGE_MAX_TOKENS` | `2048` | Max tokens per LLM call |
| `SKILLFORGE_TEMPERATURE` | `0.2` | Sampling temperature |
| `SKILLFORGE_ROUNDS` | `2` | Max validate/optimize iterations |

## Project layout

```
skillforge/
├── config.py          # Config dataclass; reads env vars once at call time
├── models.py          # Trace, Skill, SkillExample dataclasses
├── llm.py             # Thin OpenAI-compatible client; raises LLMError on failure
├── harness.py         # Pipeline orchestrator
├── core/
│   ├── observer.py    # Stage 1: intent extraction
│   ├── synthesizer.py # Stage 2: skill synthesis
│   ├── validator.py   # Stage 3: execution + LLM judge scoring
│   └── optimizer.py   # Stage 4: prompt rewriting
├── registry/
│   ├── store.py       # SQLite registry (~/.skillforge/skills.db)
│   └── export.py      # HuggingFace dataset export
└── cli/
    └── main.py        # Click CLI
tests/
examples/
└── traces/            # Sample trace files
```

## Tests

```bash
pytest
```

The test suite covers models, registry, and harness parsing without making LLM calls.
Agent behaviour is tested separately with recorded fixtures (see `tests/fixtures/`).

## Contributing

The most useful contributions right now:

- **Trace importers** — parsers for LangSmith, LangFuse, Weave, and JSONL chat export formats
- **Example traces** — real workflows submitted to `examples/traces/` expand the default skill set
- **HuggingFace dataset** — curated, high-scoring skills exported and published for community use

Open an issue before starting significant work so we can discuss the approach.

## License

Apache 2.0
