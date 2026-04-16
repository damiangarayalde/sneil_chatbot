"""
Integration tests for the route-handler subgraph nodes: generate and tool router.

Wraps:
  - tests/test__node_subgraph__generate.scenarios.json
  - tests/test__node_subgraph__generate_tool_router.scenarios.json

Scenarios that provide mock llm_output / tool_router_mock in setup.mocks run
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
from tests.scenario_helpers import (
    load_scenarios_json,
    _resolve_node_fn,
    run_scenario,
)

_GENERATE_BUNDLE = Path(__file__).parent.parent / "test__node_subgraph__generate.scenarios.json"
_TOOL_ROUTER_BUNDLE = Path(__file__).parent.parent / "test__node_subgraph__generate_tool_router.scenarios.json"

_generate_obj, _generate_scenarios, _generate_meta = load_scenarios_json(_GENERATE_BUNDLE)
_tool_router_obj, _tool_router_scenarios, _tool_router_meta = load_scenarios_json(_TOOL_ROUTER_BUNDLE)


class TestGenerateNode:
    @pytest.mark.parametrize("scenario", bundle_params(_generate_scenarios))
    def test_scenario(self, scenario: dict) -> None:
        node_fn = _resolve_node_fn(_generate_obj, _generate_meta, scenario)
        run_scenario(node_fn, scenario, _generate_meta)


class TestToolRouterNode:
    @pytest.mark.parametrize("scenario", bundle_params(_tool_router_scenarios))
    def test_scenario(self, scenario: dict) -> None:
        node_fn = _resolve_node_fn(_tool_router_obj, _tool_router_meta, scenario)
        run_scenario(node_fn, scenario, _tool_router_meta)
