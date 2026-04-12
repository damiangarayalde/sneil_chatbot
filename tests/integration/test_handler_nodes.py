"""
Integration tests for the route-handler subgraph nodes: retrieve and generate.

Wraps:
  - tests/test__node_subgraph__retrieve.scenarios.json
  - tests/test__node_subgraph__generate.scenarios.json

Scenarios that provide mock llm_output / retriever_docs in setup.mocks run
offline; others require OPENAI_API_KEY.

Run fast subset:
    pytest -m no_llm tests/integration/test_handler_nodes.py
Run with verbose output:
    pytest -s      tests/integration/test_handler_nodes.py
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

_RETRIEVE_BUNDLE = Path(__file__).parent.parent / "test__node_subgraph__retrieve.scenarios.json"
_GENERATE_BUNDLE = Path(__file__).parent.parent / "test__node_subgraph__generate.scenarios.json"

_retrieve_obj, _retrieve_scenarios, _retrieve_meta = load_scenarios_json(_RETRIEVE_BUNDLE)
_generate_obj, _generate_scenarios, _generate_meta = load_scenarios_json(_GENERATE_BUNDLE)


class TestRetrieveNode:
    @pytest.mark.parametrize("scenario", bundle_params(_retrieve_scenarios))
    def test_scenario(self, scenario: dict) -> None:
        node_fn = _resolve_node_fn(_retrieve_obj, _retrieve_meta, scenario)
        run_scenario(node_fn, scenario, _retrieve_meta)


class TestGenerateNode:
    @pytest.mark.parametrize("scenario", bundle_params(_generate_scenarios))
    def test_scenario(self, scenario: dict) -> None:
        node_fn = _resolve_node_fn(_generate_obj, _generate_meta, scenario)
        run_scenario(node_fn, scenario, _generate_meta)
