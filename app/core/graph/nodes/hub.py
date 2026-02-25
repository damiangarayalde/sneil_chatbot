from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, field_validator

from app.core.graph.routes import route_node
from app.core.graph.state import (
    ChatState,
    get_history_and_last_msg,
    get_last_msg,
    lock_route,
    reset_routing_state,
    reset_solve_state,
)
from app.core.graph.msg_heuristics_no_llm import (
    asked_for_human,
    default_clarifier,
    direct_route_from_keywords,
    escalation_message,
    is_low_info,
    route_disambiguation_question,
    wrap_with_greeting,
)
from app.core.prompts.builders import make_chat_prompt_for_route
from app.core.utils import get_routes, init_llm, is_valid_route


ALLOWED_ROUTES = set(get_routes())


class ClassifierOutput(BaseModel):
    """Structured output model used to enforce the classifier's response shape."""

    estimated_route: Literal["TPMS", "AA", "CLIMATIZADOR"] = Field(
        ...,
        description="Clasifica el tipo de mensaje como un route_id válido (ej: TPMS, AA, CLIMATIZADOR).",
    )
    confidence: float = Field(..., ge=0, le=1)
    clarifying_question: Optional[str] = Field(
        None,
        description="If confidence is low, ask ONE short clarifying question that would most improve routing.",
    )

    @field_validator("estimated_route")
    @classmethod
    def validate_route(cls, v: str) -> str:
        v = (v or "").strip()
        if v not in ALLOWED_ROUTES:
            raise ValueError(f"Invalid estimated_route: {v}")
        return v


# --------------------------------------------------------------------------------------
# Cached resources (quick win: avoid import-time side effects & speed up)

@lru_cache(maxsize=1)
def _get_classifier_resources():
    """
    Build classifier chain lazily and cache it.
    Returns: (chain, classifier_cfg)
    """
    llm = init_llm(model="gpt-4o-mini", temperature=0)
    classifier_prompt, classifier_cfg = make_chat_prompt_for_route(
        "CLASSIFIER")
    chain = classifier_prompt | llm.with_structured_output(ClassifierOutput)
    return chain, classifier_cfg


def _classifier_cfg() -> dict:
    return _get_classifier_resources()[1]


def _classifier_chain():
    return _get_classifier_resources()[0]


def _max_routing_attempts_before_handoff() -> int:
    cfg = _classifier_cfg()
    return int(cfg.get("max_attempts_before_handoff") or 0)


def _route_lock_threshold() -> float:
    cfg = _classifier_cfg()
    return float(cfg.get("route_lock_threshold") or 0.7)


@lru_cache(maxsize=64)
def _max_solve_attempts_for_route(route_id: str) -> int:
    # Reads route cfg to store threshold in state when locking
    _prompt, cfg = make_chat_prompt_for_route(route_id)
    return int(cfg.get("max_attempts_before_handoff") or 0)


# --------------------------------------------------------------------------------------
# High-level nodes

def node__clarify(state: ChatState) -> ChatState:
    """
    Clarify when user msg is too short / low info.
    Works for both:
      - before routing is locked (generic clarifier)
      - after routing is locked (route-specific disambiguation question)
    """
    locked = state.get("locked_route")
    if is_valid_route(locked):
        q = route_disambiguation_question(locked)
        text = wrap_with_greeting(q)
    else:
        text = default_clarifier()

    return {
        "messages": [AIMessage(content=text)],
        "retrieved": None,
    }


def node__handoff(state: ChatState) -> ChatState:
    """
    Handoff when:
      - user asks for human
      - routing attempts exceeded
      - solve attempts exceeded for locked route

    Quick-win: use centralized state reset helpers so resets don't drift.
    """
    msg = (
        "Disculpá — para no hacerte perder tiempo, mejor lo pasamos con una persona.\n\n"
        f"{escalation_message()}"
    )
    return {
        "messages": [AIMessage(content=msg)],
        "escalated_to_human": True,
        **reset_routing_state(),
        **reset_solve_state(),
    }


def route_from_start_precheck(state: ChatState) -> str:
    """
    START router:
      1) handoff (human request / attempts exceeded)
      2) clarify (low info)
      3) if locked => route handler
      4) else => hub
    """
    last_msg = get_last_msg(state.get("messages") or [])

    routing_attempts = int(state.get("routing_attempts") or 0)
    solve_attempts = int(state.get("solve_attempts") or 0)
    max_solve_attempts = int(state.get("max_solve_attempts") or 0)

    locked = state.get("locked_route")

    # 1) explicit human request always wins
    if asked_for_human(last_msg):
        return "handoff"

    # 2) routing attempts cap
    max_routing = _max_routing_attempts_before_handoff()
    if max_routing and routing_attempts >= max_routing:
        return "handoff"

    # 3) solve attempts cap (only if locked)
    if is_valid_route(locked) and max_solve_attempts and solve_attempts >= max_solve_attempts:
        return "handoff"

    # 4) low-info clarify (generic or route-specific)
    if is_low_info(last_msg):
        return "clarify"

    # 5) locked => handler, else hub
    if is_valid_route(locked):
        return route_node(locked)

    return "hub"


# --------------------------------------------------------------------------------------
# Hub classifier node (ONLY classification + route lock)

def node__classify_user_intent(state: ChatState) -> ChatState:
    """Hub / classifier node.

    Rules (in order):
    1) If already locked -> pass-through
    2) If cheap keyword routing works -> lock route (no LLM)
    3) Else call LLM:
        - low confidence -> greet + clarifier question (routing_attempts += 1)
        - high confidence -> lock route
    """
    locked = state.get("locked_route")
    if is_valid_route(locked):
        return {}

    prior_messages, last_message = get_history_and_last_msg(
        state.get("messages") or [])
    last_message = last_message or ""
    routing_attempts = int(state.get("routing_attempts") or 0)

    # cheap direct routing (keywords, route mentions, etc.)
    direct = direct_route_from_keywords(last_message, ALLOWED_ROUTES)
    if direct:
        return lock_route(
            direct,
            confidence=1.0,
            max_solve_attempts=_max_solve_attempts_for_route(direct),
        )

    # LLM classifier
    meta_text = f"routing_attempts={routing_attempts}\n"
    result = _classifier_chain().invoke(
        {
            "user_text": last_message,
            "history": prior_messages,
            "context": "",
            "meta": meta_text,
        }
    )

    has_clarifier = bool((result.clarifying_question or "").strip())
    if (not has_clarifier) and float(result.confidence) >= _route_lock_threshold():
        route_id = result.estimated_route
        updates = lock_route(
            route_id,
            confidence=float(result.confidence),
            max_solve_attempts=_max_solve_attempts_for_route(route_id),
        )
        return updates

    # low confidence => ask a routing question (counts as routing_attempt)
    question = (result.clarifying_question or "").strip() or route_disambiguation_question(
        result.estimated_route
    )
    return {
        "confidence": float(result.confidence),
        "estimated_route": result.estimated_route,
        "routing_attempts": routing_attempts + 1,
        "messages": [AIMessage(content=wrap_with_greeting(question))],
    }
