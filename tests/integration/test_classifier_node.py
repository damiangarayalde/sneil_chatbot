"""
Integration tests for the route-classifier node.

Each scenario in test__node_route_classifier.scenarios.json becomes one
parametrized test case.  Scenarios tagged no_llm run fully offline; scenarios
tagged llm require OPENAI_API_KEY.

Run fast (offline) subset:
    pytest -m "no_llm" tests/integration/test_classifier_node.py
Run with verbose per-turn output (mirrors the CLI runner):
    pytest -s   tests/integration/test_classifier_node.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.conftest import bundle_params
from tests.run__node_or_graph__test import (
    load_scenarios_json,
    _resolve_node_fn,
    run_scenario,
)

_BUNDLE = Path(__file__).parent.parent / "test__node_route_classifier.scenarios.json"
_obj, _scenarios, _bundle_meta = load_scenarios_json(_BUNDLE)


@pytest.mark.parametrize("scenario", bundle_params(_scenarios))
def test_scenario(scenario: dict) -> None:
    node_fn = _resolve_node_fn(_obj, _bundle_meta, scenario)
    run_scenario(node_fn, scenario, _bundle_meta)
