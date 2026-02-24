from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import AIMessage

from app.core.graph.state import ChatState, get_history_and_last_msg
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
max_attempts_before_handoff = int(
    classifier_cfg.get("max_attempts_before_handoff") or 0)
route_lock_threshold = float(classifier_cfg.get("route_lock_threshold") or 0.7)


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
        return {}

    prior_messages, last_message = get_history_and_last_msg(
        state.get("messages") or [])
    last_message = last_message or ""

    routing_attempts = int(state.get("routing_attempts") or 0)

    # (still here in Step 1) escalation
    if routing_attempts >= max_attempts_before_handoff or asked_for_human(last_message):
        return {
            "escalated_to_human": True,
            "routing_attempts": 0,
            "solve_attempts": 0,
            "attempts": 0,  # legacy
            "messages": [AIMessage(content=escalation_message())],
        }

    # cheap direct routing
    direct = direct_route_from_keywords(last_message, ALLOWED_ROUTES)
    if direct:
        # For now, don’t set max_solve_attempts yet (we’ll do it in Step 3 cleanly)
        return {
            "confidence": 1.0,
            "estimated_route": direct,
            "locked_route": direct,
            "routing_attempts": 0,
            "solve_attempts": 0,
            "attempts": 0,  # legacy
        }

    # low-info => don't lock
    if is_low_info(last_message):
        return {
            "confidence": 0.0,
            "routing_attempts": routing_attempts,  # low info doesn't count
            "attempts": routing_attempts,          # legacy
            "messages": [AIMessage(content=default_clarifier())],
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
        return {
            "confidence": float(result.confidence),
            "estimated_route": result.estimated_route,
            "locked_route": result.estimated_route,
            "routing_attempts": 0,
            "solve_attempts": 0,
            "attempts": 0,  # legacy
        }

    question = (result.clarifying_question or "").strip() or route_disambiguation_question(
        result.estimated_route
    )
    return {
        "confidence": float(result.confidence),
        "estimated_route": result.estimated_route,
        "routing_attempts": routing_attempts + 1,
        "attempts": routing_attempts + 1,  # legacy
        "messages": [AIMessage(content=wrap_with_greeting(question))],
    }
