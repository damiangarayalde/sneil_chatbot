"""Scenario-driven smoke tests for hub routing rules (REAL LLM)."""

from __future__ import annotations

from pathlib import Path
from app.core.graph.nodes.hub import node__classify_user_intent
from utility_to_run_scenarios import load_scenarios_json, run_all


def main() -> None:
    # Convention: keep the scenarios file next to this script.
    path_to_scenarios = Path(__file__).with_suffix(".scenarios.json")
    scenarios = load_scenarios_json(path_to_scenarios)
    run_all(node__classify_user_intent, scenarios)


if __name__ == "__main__":
    main()
