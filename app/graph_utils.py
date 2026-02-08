from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional

from app.core.types import ChatState
from langchain_core.messages import BaseMessage

# ---- Logging config ----
_MAX_STR = 100   # max chars shown for any string
_MAX_KVS = 6     # max extra key/values shown per line
_PHASES = ["triage", "handling", "closed"]

_step_counter = 0


def _bump_step() -> int:
    global _step_counter
    _step_counter += 1
    return _step_counter


def _ansi_enabled() -> bool:
    # Respect NO_COLOR (https://no-color.org/)
    if os.getenv("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


ANSI = _ansi_enabled()
RESET = "\033[0m" if ANSI else ""
BOLD = "\033[1m" if ANSI else ""
DIM = "\033[90m" if ANSI else ""  # grey/dim


def _trim(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).replace("\n", " ").strip()
    return s if len(s) <= _MAX_STR else (s[: _MAX_STR - 1] + "…")


def _style_current(phase: str) -> str:
    # Current phase in bold (or uppercase if no ANSI)
    if ANSI:
        return f"{BOLD}{phase}{RESET}"
    return phase.upper()


def _style_inactive(phase: str) -> str:
    # Non-current phases greyed out (or plain if no ANSI)
    if ANSI:
        return f"{DIM}{phase}{RESET}"
    return phase


def _render_phase_bar(in_phase: Optional[str], out_phase: Optional[str]) -> str:
    """Render a single-line phase bar.

    - If unchanged: [ triage | handling | closed ] with current highlighted.
    - If changed:   [ triage > handling | closed ] or [ triage | handling > closed ].
    """
    ip = in_phase or "triage"
    op = out_phase or ip

    # If no phase change
    if ip == op:
        parts = []
        for p in _PHASES:
            parts.append(_style_current(p) if p == ip else _style_inactive(p))
        return "[ " + " | ".join(parts) + " ]"

    # Phase changed: show "ip > op" grouped once, plus the remaining phase.
    # We expect mostly triage->handling or handling->closed. For anything else, still render sanely.
    parts: List[str] = []
    used = set()

    for p in _PHASES:
        if p == ip:
            # if op is adjacent or anywhere, show transition token once
            trans = f"{_style_inactive(ip)} > {_style_current(op)}"
            parts.append(trans)
            used.add(ip)
            used.add(op)
        elif p == op:
            # already included in the transition segment
            continue
        else:
            parts.append(_style_inactive(p))

    return "[ " + " | ".join(parts) + " ]"


def _node_segment(node_name: str, next_node: Optional[str]) -> str:
    if next_node and next_node != node_name:
        return f"Node[{node_name} -> {next_node}]"
    return f"Node[{node_name}]"


def _non_empty(v: Any) -> bool:
    return not (v is None or v == "" or v == [] or v == {})


def _fmt_kv(k: str, v: Any) -> str:
    if isinstance(v, float):
        return f"{k}={v:.3f}"
    if isinstance(v, str):
        return f"{k}={_trim(v)}"
    return f"{k}={_trim(v)}"


def _pick_tail_fields(merged: Dict[str, Any]) -> List[str]:
    # Keep this intentionally small and “signal-heavy”.
    # (No messages, no giant summaries.)
    preferred = [
        "Locked_route",   # we will map from locked_route
        "confidence",
        "routing_attempts",
        "triage_question",
    ]
    return preferred


def _delta_normalize_messages(input_state: ChatState, result: Dict[str, Any]) -> Dict[str, Any]:
    """If a runnable/subgraph returns a FULL messages list, convert to delta to avoid duplication."""
    if "messages" not in result:
        return result

    in_msgs = input_state.get("messages") or []
    out_msgs = result.get("messages") or []

    if not isinstance(in_msgs, list) or not isinstance(out_msgs, list):
        return result
    if not out_msgs:
        return result
    if not all(isinstance(m, BaseMessage) for m in out_msgs):
        return result

    # Heuristic: if the output starts with the input history, treat it as full-state return
    if len(out_msgs) >= len(in_msgs):
        try:
            prefix_matches = True
            for i in range(len(in_msgs)):
                if out_msgs[i] is not in_msgs[i] and repr(out_msgs[i]) != repr(in_msgs[i]):
                    prefix_matches = False
                    break
            if prefix_matches:
                result = dict(result)
                result["messages"] = out_msgs[len(in_msgs):]
        except Exception:
            return result

    return result


def wrap_node(name: str, fn: Callable[[ChatState], ChatState]) -> Callable[[ChatState], ChatState]:
    """Wrap a node (or compiled subgraph) and print a one-line flow log per execution."""

    def wrapper(state: ChatState) -> ChatState:
        step = _bump_step()

        in_phase = state.get("phase")
        in_next = state.get("next")

        # Execute
        if hasattr(fn, "invoke"):
            result = fn.invoke(state)  # type: ignore[attr-defined]
        else:
            result = fn(state)

        # ✅ If this is a runnable/subgraph that returned full state, keep only delta messages
        if isinstance(result, dict):
            result = _delta_normalize_messages(state, result)

        # Determine out_phase/out_next (assume unchanged if not provided)
        out_phase = in_phase
        out_next = in_next
        if isinstance(result, dict):
            if _non_empty(result.get("phase")):
                out_phase = result.get("phase")
            if _non_empty(result.get("next")):
                out_next = result.get("next")

        # Render phase and node segments
        phase_bar = _render_phase_bar(in_phase, out_phase)
        node_seg = _node_segment(
            name, out_next if isinstance(out_next, str) else None)

        # Merge state+result to display final meaningful fields
        merged: Dict[str, Any] = dict(state)
        if isinstance(result, dict):
            merged.update(result)

        # Build tail: skip None/empty; skip legacy fields; skip messages always
        tail_parts: List[str] = []

        locked = merged.get("locked_route")
        if _non_empty(locked):
            tail_parts.append(f"Locked_route={locked}")

        conf = merged.get("confidence")
        if isinstance(conf, (float, int)):
            tail_parts.append(f"confidence={float(conf):.3f}")

        ra = merged.get("routing_attempts")
        if isinstance(ra, int) and ra != 0:
            tail_parts.append(f"routing_attempts={ra}")

        tq = merged.get("triage_question")
        if _non_empty(tq):
            tail_parts.append(f"triage_question='{_trim(tq)}'")

        # (Optional) keep a small space for other future keys if you want,
        # but for now we stay minimal.
        tail = ("  " + " ".join(tail_parts[:_MAX_KVS])) if tail_parts else ""

        print(f"● {step:03d} {phase_bar} {node_seg}{tail}\n")
        return result

    return wrapper
