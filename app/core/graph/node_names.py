from __future__ import annotations


def route_node(route: str) -> str:
    """Canonical node name for a route handler."""
    return f"handle__{route}"


def end_turn_node() -> str:
    """Single place to define the end-of-turn node name."""
    return "end_of_turn"
