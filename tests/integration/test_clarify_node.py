"""
Integration tests for the clarify node.

Wraps tests/test__node_clarify.scenarios.json.
All scenarios are tagged no_llm — fully offline.

Run:
    pytest tests/integration/test_clarify_node.py -v
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

_BUNDLE = Path(__file__).parent.parent / "test__node_clarify.scenarios.json"
_obj, _scenarios, _bundle_meta = load_scenarios_json(_BUNDLE)


@pytest.mark.parametrize("scenario", bundle_params(_scenarios))
def test_scenario(scenario: dict) -> None:
    node_fn = _resolve_node_fn(_obj, _bundle_meta, scenario)
    run_scenario(node_fn, scenario, _bundle_meta)
