"""
Integration-tier conftest.

Provides `_bundle_params` — a helper used by every integration test file to
turn a scenario JSON bundle into a list of `pytest.param` objects with the
correct marks applied automatically from each scenario's `tags` array.

Usage in a test file:
    from tests.integration.conftest import bundle_params

    _BUNDLE = Path(__file__).parent.parent / "test__node_route_classifier.scenarios.json"
    _obj, _scenarios, _bundle_meta = load_scenarios_json(_BUNDLE)

    @pytest.mark.parametrize("scenario", bundle_params(_scenarios))
    def test_scenario(scenario): ...
"""
from __future__ import annotations

from typing import Any

import pytest


def bundle_params(scenarios: list[dict[str, Any]]) -> list[pytest.param]:
    """Convert a list of scenario dicts into parametrize-ready pytest.param objects.

    Marks applied automatically:
    - always:   pytest.mark.integration
    - if "no_llm" in tags: pytest.mark.no_llm  (else: pytest.mark.llm)
    """
    params = []
    for s in scenarios:
        tags = s.get("tags") or []
        marks = [pytest.mark.integration]
        if "no_llm" in tags:
            marks.append(pytest.mark.no_llm)
        else:
            marks.append(pytest.mark.llm)
        params.append(pytest.param(s, marks=marks, id=s.get("id", "unknown")))
    return params
