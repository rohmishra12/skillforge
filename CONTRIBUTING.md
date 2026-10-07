# Contributing to SkillForge

## Before you start

Open an issue before beginning significant work. It avoids duplicated effort and gives
us a chance to discuss whether the approach fits the project's direction.

For small fixes (typos, minor bugs, documentation) a PR without a prior issue is fine.

## Development setup

```bash
git clone https://github.com/rohmishra12/skillforge
cd skillforge
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Set your API key before running any end-to-end tests:

```bash
export SKILLFORGE_API_KEY=sk-...
```

## Running tests

```bash
pytest                     # all tests
pytest tests/test_models.py -v   # specific file
```

The unit tests in `tests/` make no LLM calls and should run without an API key.

## Code style

```bash
ruff check .        # lint
ruff format .       # format
mypy skillforge/    # type check
```

CI runs all three. PRs that fail CI won't be merged.

## What to contribute

**High value:**

- Trace importers — parsers for LangSmith, LangFuse, Weave, and plain JSONL chat exports.
  Add them to `skillforge/importers/` with a corresponding test using a small fixture file.
- Example traces — real workflows in `examples/traces/`. The filename becomes the default
  skill name, so make it descriptive (`draft_incident_report.json`, not `trace2.json`).
- HuggingFace dataset — skills with a validation score ≥ 0.8, exported and submitted as a PR
  to the companion dataset repository.

**Lower priority:**

- Web UI (Gradio/Streamlit) — install with `pip install skillforge[web]`. Keep it in `skillforge/web/`.
- Additional LLM backends — the `llm.py` client already supports any OpenAI-compatible endpoint,
  so this is rarely needed.

## Pull request checklist

- [ ] Tests pass (`pytest`)
- [ ] No new linting errors (`ruff check .`)
- [ ] New behaviour is covered by at least one test
- [ ] Commit messages are imperative, present tense: "add LangSmith importer" not "added importer"

## Versioning

SkillForge follows [Semantic Versioning](https://semver.org). The `Skill.version` field uses
the same scheme for individual skills — the optimizer increments the patch number on each
rewrite pass.
