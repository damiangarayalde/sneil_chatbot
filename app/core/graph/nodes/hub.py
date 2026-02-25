from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import AIMessage

from app.core.graph.state import ChatState, get_history_and_last_msg, get_last_msg
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
from app.core.utils import init_llm, get_routes, is_valid_route
from app.core.graph.routes import route_node


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


# Initialize LLM and Prompt
llm = init_llm(model="gpt-4o-mini", temperature=0)
classifier_prompt, classifier_cfg = make_chat_prompt_for_route("CLASSIFIER")
chain = classifier_prompt | llm.with_structured_output(ClassifierOutput)

# Max number of iterative solution attempts before we suggest human handoff.
max_routing_attempts_before_handoff = int(
    classifier_cfg.get("max_attempts_before_handoff") or 0)
route_lock_threshold = float(classifier_cfg.get("route_lock_threshold") or 0.7)


def _max_solve_attempts_for_route(route_id: str) -> int:
    # Reads route cfg to store threshold in state when locking
    # improve evitando built promp only to get to config
    _prompt, cfg = make_chat_prompt_for_route(route_id)
    return int(cfg.get("max_attempts_before_handoff") or 0)


# -------------------------------------------------------------------------
# High-level nodes (NEW)
# -------------------------------------------------------------------------

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
        # default_clarifier already designed for triage style
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
    Resets locks and attempts. 

    the answer should be different if the user requested for assistance or we have exceeded the attempts
    """
    msg = (
        "Disculpá — para no hacerte perder tiempo, mejor lo pasamos con una persona.\n\n"
        f"{escalation_message()}"
    )
    return {
        "messages": [AIMessage(content=msg)],
        "escalated_to_human": True,
        "locked_route": None,
        "confidence": 0,
        "estimated_route": None,
        "retrieved": None,
        "routing_attempts": 0,
        "solve_attempts": 0,
        "max_solve_attempts": None,
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
    if max_routing_attempts_before_handoff and routing_attempts >= max_routing_attempts_before_handoff:
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


# -------------------------------------------------------------------------
# Hub classifier node (now ONLY classification + route lock)
# -------------------------------------------------------------------------

def node__classify_user_intent(state: ChatState) -> ChatState:
    """Hub / classifier node.

    Rules (in order):
    1) If already locked -> pass-through
    2) If (attempts >= MAX) OR user asks for human -> escalate
    3) If (support/sale) + single route mention -> lock route (no LLM)
    4) If low-info -> greet + default clarifier question
    5) Else call LLM:
        - low confidence -> greet + clarifier question
        - high confidence -> lock route
    """

    locked = state.get("locked_route")
    if is_valid_route(locked):

        print(" AT HUB:   LOCKED ROUTE , XXXXXXXXXX")
        return {}

    prior_messages, last_message = get_history_and_last_msg(
        state.get("messages") or [])
    last_message = last_message or ""

    routing_attempts = int(state.get("routing_attempts") or 0)

    # cheap direct routing
    direct = direct_route_from_keywords(last_message, ALLOWED_ROUTES)
    if direct:
        return {
            "confidence": 1.0,
            "estimated_route": direct,
            "locked_route": direct,
            "routing_attempts": 0,
            "solve_attempts": 0,
            "attempts": 0,  # legacy
            "max_solve_attempts": _max_solve_attempts_for_route(direct),
        }

    # LLM classifier
    meta_text = f"routing_attempts={routing_attempts}\n"
    result = chain.invoke(
        {
            "user_text": last_message,
            "history": prior_messages,
            "context": "",
            "meta": meta_text,
        }
    )

    has_clarifier = bool((result.clarifying_question or "").strip())
    if (not has_clarifier) and float(result.confidence) >= route_lock_threshold:
        route_id = result.estimated_route
        return {
            "confidence": float(result.confidence),
            "estimated_route": route_id,
            "locked_route": route_id,
            "routing_attempts": 0,
            "solve_attempts": 0,
            "max_solve_attempts": _max_solve_attempts_for_route(route_id),
        }

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
