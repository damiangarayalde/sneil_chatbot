"""
Integration tests for the full route subgraph (multi-node, compiled).

Wraps tests/test__route_subgraph.scenarios.json.

This bundle uses a different schema from the single-node bundles — it is a raw
list of scenarios (no `node_ref` wrapper) where each scenario carries its own
`route_id`.  The test compiles the route subgraph directly and invokes it,
applying the same assertion helpers used by the node-level tests.

Most scenarios here call the generate node and therefore require OPENAI_API_KEY.
Mark a scenario no_llm by adding "tags": ["no_llm"] to its JSON entry.

Run:
    pytest -m "integration and llm" tests/integration/test_route_subgraph.py -v
    pytest -s -k "Scenario__1"    tests/integration/test_route_subgraph.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from tests.integration.conftest import bundle_params
from tests.scenario_helpers import (
    _apply_patch,
    _assert_expect,
    _last_ai_message,
    _matches_expect,
)

_BUNDLE_PATH = Path(__file__).parent.parent / "test__route_subgraph.scenarios.json"
_scenarios: list[dict[str, Any]] = json.loads(_BUNDLE_PATH.read_text(encoding="utf-8"))


def _run_subgraph_scenario(scenario: dict[str, Any]) -> None:
    """Compile the route subgraph for the scenario's route_id and run all turns."""
    from app.core.graph.route_handler.factory_and_nodes import make_route_subgraph

    route_id: str = scenario.get("route_id") or scenario.get("setup", {}).get("route_id", "")
    if not route_id:
        raise ValueError(f"Scenario {scenario.get('id')} has no route_id")

    compiled = make_route_subgraph(route_id)

    state: dict[str, Any] = {"messages": [], "locked_route": route_id}
    state.update((scenario.get("setup") or {}).get("initial_state") or {})

    print(f"\n{'=' * 100}")
    print(f"{scenario.get('id')}: {scenario.get('title')}")

    for i, turn in enumerate(scenario.get("turns") or [], start=1):
        user_msg = turn["user_msg"]
        state.setdefault("messages", []).append(HumanMessage(content=user_msg))

        patch = compiled.invoke(state) or {}
        state = _apply_patch(dict(state), patch)

        last_ai = _last_ai_message(state.get("messages") or [])
        print(f"\n  Turn {i} > user: {repr(user_msg)}")
        if last_ai:
            print(f"  assistant: {last_ai}")
        print(f"  state: locked={state.get('locked_route')}  "
              f"escalated={state.get('escalated_to_human')}  "
              f"attempts={state.get('solve_attempts')}")

        exp = turn.get("expect")
        if not exp:
            continue

        if isinstance(exp, dict) and "one_of" in exp:
            options = exp["one_of"] or []
            if not any(_matches_expect(state, opt) for opt in options):
                raise AssertionError(
                    f"No expectation in one_of matched\n"
                    f" one_of: {options}\n"
                    f" got: {state}"
                )
        else:
            _assert_expect(state, exp)


@pytest.mark.parametrize("scenario", bundle_params(_scenarios))
def test_scenario(scenario: dict) -> None:
    _run_subgraph_scenario(scenario)
