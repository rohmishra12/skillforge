# Test Fixtures

This directory contains recorded LLM responses for testing the four synthesis agents without requiring a live API key.

## Structure

- `observer/` - Observer agent fixtures
- `synthesizer/` - Synthesizer agent fixtures  
- `validator/` - Validator agent fixtures
- `optimizer/` - Optimizer agent fixtures

Each fixture is a JSON file containing:
- `request` - The messages sent to the LLM
- `response` - The exact response received
- `metadata` - Test context (model, temperature, etc.)

## Recording New Fixtures

To record new fixtures during development:

```bash
SKILLFORGE_RECORD_FIXTURES=1 pytest tests/test_agents.py::test_observer -v
```

This will capture the real API responses and save them as fixtures for future test runs.

## Usage

The test suite automatically uses fixtures when they exist, falling back to live API calls only when `SKILLFORGE_API_KEY` is set and no matching fixture exists.