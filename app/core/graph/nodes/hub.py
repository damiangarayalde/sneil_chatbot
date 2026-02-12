from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from app.core.graph.state import ChatState, get_history_and_last_msg
from app.core.prompts.builders import make_chat_prompt_for_route
from app.core.utils import init_llm, get_routes, is_valid_route
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

ALLOWED_ROUTES = set(get_routes())

# Initialize the chat model used by the application
llm = init_llm(model="gpt-4o-mini", temperature=0)

# --- Step 3A routing configuration (keep here for now; can move later) ---
ROUTE_LOCK_THRESHOLD = 0.75
MAX_ROUTING_ATTEMPTS = 3

# How much history to include in the classifier prompt (keep small for cost)
CLASSIFIER_HISTORY_MAX_MESSAGES = 10
CLASSIFIER_HISTORY_MAX_CHARS = 2500


class UserIntentClassifier_output_format(BaseModel):
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


# Treat these as "low information" replies that should NOT lock a route
LOW_INFO_MSGS = {
    "hola", "buenas", "buen día", "buen dia", "buenas tardes", "buenas noches",
    "ok", "oka", "dale", "listo", "joya", "perfecto", "bien",
    "si", "sí", "no", "seguro", "claro", "gracias", "👍", "👌"
}


def _is_low_info(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if t in LOW_INFO_MSGS:
        return True
    # very short acknowledgements are usually not enough to lock
    if len(t) <= 3:
        return True
    return False


def _route_disambiguation_question(route_guess: str) -> str:
    # Route-specific fallback if you already have a good guess
    if route_guess == "TPMS":
        return "¿Es un problema de lectura de presión/temperatura, sensores que no aparecen, o instalación/configuración del TPMS?"
    if route_guess == "AA":
        return "¿Es sobre instalación/potencia/consumo del AA, o sobre rendimiento (no enfría / no calienta / caudal)?"
    if route_guess == "CLIMATIZADOR":
        return "¿Es sobre instalación/uso del climatizador o sobre rendimiento/consumo/ruidos?"
    return "¿Podés decirme qué producto es y cuál es el problema principal?"


# Build the classifier prompt from config (prompt_file under CLASSIFIER)
classifier_prompt, _ = make_chat_prompt_for_route("CLASSIFIER")


def node__classify_user_intent(state: ChatState) -> ChatState:
    # PASS-THROUGH: if a route is already locked, do nothing.
    # Graph edges will send execution to the correct handler.
    locked = state.get("locked_route")
    if is_valid_route(locked):
        return {}

    # Use the last user message to classify the route and confidence
    prior_messages, last_message = get_history_and_last_msg(
        state.get("messages") or [])

    # --- HARD GUARD: low-info user replies should NOT lock a route
    if _is_low_info(last_message):
        current_guess = state.get("estimated_route") or None
        q = _route_disambiguation_question(current_guess)
        return {
            "confidence": min(float(state.get("confidence", 0)), 0),
            "routing_attempts": state.get("routing_attempts", 0) + 1,
            "messages": [AIMessage(content=q)],
        }

    # If the msg pass the basic low-info filter, proceed with normal classification flow. This allows for some borderline cases to be classified based on their content, while still preventing obviously unhelpful messages from locking routes.
    classifier_llm = llm.with_structured_output(
        UserIntentClassifier_output_format)

    # filtered = []
    # for m in prior_messages:
    #     if isinstance(m, (HumanMessage, AIMessage)):
    #         filtered.append(m)

    # history: list[BaseMessage] = filtered[-CLASSIFIER_HISTORY_MAX_MESSAGES:]

    # # Cap total history size by dropping oldest messages (keeps type=list[BaseMessage]).
    # total_chars = sum(len(getattr(m, "content", "") or "") for m in history)
    # while history and total_chars > CLASSIFIER_HISTORY_MAX_CHARS:
    #     dropped = history.pop(0)
    #     total_chars -= len(getattr(dropped, "content", "") or "")

    attempts = int(state.get("routing_attempts") or 0)

    # Build internal metadata for the prompt (kept as SYSTEM, not HUMAN)
    meta_text = (
        f"routing_attempts={attempts}\n"
    )

    fmt_kwargs = {
        "user_text": last_message,
        # MUST be list[BaseMessage],.. was history before
        "history": prior_messages,
        "context": "",  # this is used on paths for rag/catalog data
        "meta": meta_text,
    }

    chain = classifier_prompt | classifier_llm
    result = chain.invoke(fmt_kwargs)

    # If confidence is high enough OR attempts exceeded, lock route and proceed
    if float(result.confidence) >= ROUTE_LOCK_THRESHOLD or attempts >= MAX_ROUTING_ATTEMPTS:
        return {
            "confidence": float(result.confidence),
            "locked_route": result.estimated_route,
            "routing_attempts": 0,  # reset
        }

    # Low confidence: ask one clarifying question (must not be generic)
    question = (result.clarifying_question or "").strip(
    ) or _route_disambiguation_question(result.estimated_route)
    return {
        "confidence": float(result.confidence),
        "routing_attempts": attempts + 1,
        "messages": [AIMessage(content=question)],
    }
