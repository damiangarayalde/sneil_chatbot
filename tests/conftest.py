"""
Root conftest — registers custom pytest markers so `-m no_llm` etc. work
without warnings, and provides the `scenario_runner` utility used by all
integration test files.
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register project-specific markers (already declared in pyproject.toml,
    but registering here avoids 'unknown mark' warnings during collection)."""
    config.addinivalue_line("markers", "unit: pure Python, no LLM, no I/O")
    config.addinivalue_line("markers", "integration: node-level, may call LLM")
    config.addinivalue_line("markers", "e2e: full graph, multi-turn")
    config.addinivalue_line("markers", "no_llm: scenario uses only heuristics — safe offline")
    config.addinivalue_line("markers", "llm: scenario requires a real OpenAI call")
