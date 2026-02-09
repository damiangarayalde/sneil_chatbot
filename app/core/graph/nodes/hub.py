from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from app.core.graph.state import ChatState
from app.core.prompts.builders import make_chat_prompt_for_route
from app.core.utils import init_llm, get_routes
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from app.core.utils import is_valid_route

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

# delete below -------#


def _role_label(msg: BaseMessage) -> str:
    # Keep it simple; we only need enough to help routing.
    if isinstance(msg, HumanMessage):
        return "USER"
    return "ASSISTANT"


def _format_history(messages: list[BaseMessage]) -> str:
    """
    Build a compact, role-tagged history string.

    We include only the last N messages (excluding the latest user message,
    which is provided separately as {user_text}), and cap total chars to keep costs predictable.
    """
    if not messages:
        return ""

    tail = messages[-CLASSIFIER_HISTORY_MAX_MESSAGES:]
    lines = [f"{_role_label(m)}: {m.content}" for m in tail if getattr(
        m, "content", None)]
    history = "\n".join(lines)

    if len(history) > CLASSIFIER_HISTORY_MAX_CHARS:
        history = history[-CLASSIFIER_HISTORY_MAX_CHARS:]
        history = "…(recortado)\n" + history

    return history
# ------- delete above -------#


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
    last_message = state["messages"][-1].content

    # --- HARD GUARD: low-info user replies should NOT lock a route
    if _is_low_info(last_message):
        current_guess = state.get("estimated_route") or None
        q = _route_disambiguation_question(current_guess)
        return {
            "confidence": min(float(state.get("confidence", 0)), 0),
            "routing_attempts": state.get("routing_attempts", 0) + 1,
            "triage_question": q,
            "messages": [AIMessage(content=q)],
        }

    # If the msg pass the basic low-info filter, proceed with normal classification flow. This allows for some borderline cases to be classified based on their content, while still preventing obviously unhelpful messages from locking routes.
    classifier_llm = llm.with_structured_output(
        UserIntentClassifier_output_format)

    # Build history as list[BaseMessage] (best practice). Exclude current user message.
    prior_messages = state.get("messages", [])[:-1]
    # prior_messages: list[BaseMessage] = state.get("messages", [])[:-1]

    # Legacy string-history formatter kept above for reference:
    # history = _format_history(prior_messages)

    filtered = []
    for m in prior_messages:
        if isinstance(m, (HumanMessage, AIMessage)):
            filtered.append(m)

    history: list[BaseMessage] = filtered[-CLASSIFIER_HISTORY_MAX_MESSAGES:]

    # Cap total history size by dropping oldest messages (keeps type=list[BaseMessage]).
    total_chars = sum(len(getattr(m, "content", "") or "") for m in history)
    while history and total_chars > CLASSIFIER_HISTORY_MAX_CHARS:
        dropped = history.pop(0)
        total_chars -= len(getattr(dropped, "content", "") or "")

    # Try to extract sender/phone info from the last message metadata if present
    last_msg_obj = state["messages"][-1]
    from_val = ""
    try:
        meta = getattr(last_msg_obj, "metadata", None)
        if isinstance(meta, dict):
            from_val = meta.get("from", "") or meta.get("sender", "")
    except Exception:
        from_val = ""

    attempts = int(state.get("routing_attempts") or 0)
    triage_summary = (state.get("triage_summary") or "").strip()

    fmt_kwargs = {
        "user_text": last_message,
        "text": last_message,  # kept for compatibility in case prompts reference it
        "history": history,    # MUST be list[BaseMessage]
        "from": from_val,
        "routing_attempts": attempts,
        "triage_summary": triage_summary,
    }

    result = classifier_llm.invoke(
        classifier_prompt.format_messages(**fmt_kwargs))

    # If confidence is high enough OR attempts exceeded, lock route and proceed
    if float(result.confidence) >= ROUTE_LOCK_THRESHOLD or attempts >= MAX_ROUTING_ATTEMPTS:
        return {
            "confidence": float(result.confidence),
            "locked_route": result.estimated_route,
            "triage_question": None,
        }

    # Low confidence: ask one clarifying question (must not be generic)
    question = (result.clarifying_question or "").strip(
    ) or _route_disambiguation_question(result.estimated_route)
    return {
        "confidence": float(result.confidence),
        "routing_attempts": attempts + 1,
        "triage_question": question,
        "messages": [AIMessage(content=question)],
    }
