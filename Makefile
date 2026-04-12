.PHONY: test-fast test test-all install-test

# Install test dependencies
install-test:
	uv pip install -e ".[test]"

# Offline — no OPENAI_API_KEY needed. Run before every edit.
test-fast:
	uv run pytest -m "unit or no_llm" --tb=short -q

# Full integration suite — run before a commit.
# Requires OPENAI_API_KEY for llm-tagged scenarios.
test:
	uv run pytest -m "not e2e" --tb=short -q

# All tests + coverage — run before a push.
test-all:
	uv run pytest --tb=short --cov=app --cov-report=term-missing -q

# Run a specific scenario by ID with full verbose (colored) output.
# Usage: make test-scenario ID=Scenario_1
test-scenario:
	uv run pytest -s -k "$(ID)" --tb=short
