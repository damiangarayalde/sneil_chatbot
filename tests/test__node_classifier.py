"""Scenario-driven smoke tests for hub routing rules (REAL LLM).

This script stays lightweight and reusable by:
- keeping the Scenario Spec (v1)
- moving SCENARIOS into a standalone JSON file
- delegating the runner to scenario_engine_nodes.py

Notes:
- This WILL call your configured LLM for scenarios that reach the classifier chain.
"""

from __future__ import annotations

from pathlib import Path

from app.core.graph.nodes.hub import node__classify_user_intent
from utility_to_run_scenarios import load_scenarios_json, run_all


def main() -> None:
    # Convention: keep the scenarios file next to this script.
    scenarios_path = Path(__file__).with_suffix(".scenarios.json")
    scenarios = load_scenarios_json(scenarios_path)
    run_all(node__classify_user_intent, scenarios)


if __name__ == "__main__":
    main()
