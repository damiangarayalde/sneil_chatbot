from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from app.types import ChatState
from app.prompts.prompt_utils import make_chat_prompt_for_route
from app.utils import init_llm, get_routes, is_valid_route, get_classifier_cfg
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

    # handling_channel: str = Field(
    #     ...,
    #     description="Clasifica el tipo de mensaje como un route_id válido (ej: TPMS, AA, CLIMATIZADOR).",
    # )
    handling_channel: Literal["TPMS", "AA", "CLIMATIZADOR"] = Field(
        ...,
        description="Clasifica el tipo de mensaje como 'TPMS', 'AA' o 'CLIMATIZADOR'.",
    )
    confidence: float = Field(..., ge=0, le=1)

    clarifying_question: Optional[str] = Field(
        None,
        description="If confidence is low, ask ONE short clarifying question that would most improve routing.",
    )

    @field_validator("handling_channel")
    @classmethod
    def validate_route(cls, v: str) -> str:
        v = (v or "").strip()
        if v not in ALLOWED_ROUTES:
            raise ValueError(f"Invalid handling_channel: {v}")
        return v


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


# IMPORTANT: this is the key change: the human template now includes history + attempts + triage_summary.
# (No unnecessary edits beyond adding the fields you explicitly requested.)
classifier_prompt, _ = make_chat_prompt_for_route(
    "CLASSIFIER",
    (
        "Historial reciente (puede estar vacío):\n{history}\n\n"
        "Intentos de ruteo hasta ahora: {routing_attempts}\n"
        "Resumen de triage actual (puede estar vacío): {triage_summary}\n"
        "Sender (si existe): {from}\n\n"
        "Último mensaje del usuario:\n{user_text}"
    ),
)


def _fallback_clarifying_question(route_guess: str) -> str:
    # Fallback in case LLM does not return clarifying_question.
    if route_guess == "TPMS":
        return "¿Tu problema es sobre sensores TPMS que no aparecen/lecturas inestables, o sobre instalación/configuración?"
    if route_guess == "AA":
        return "¿Es sobre aire acondicionado (AA) instalación/potencia/consumo, o un problema de rendimiento (enfriamiento/calefacción)?"
    if route_guess == "CLIMATIZADOR":
        return "¿Es sobre instalación/uso del climatizador o sobre rendimiento/consumo/ruidos?"
    return "¿Podés decirme qué producto es y cuál es el problema principal?"


# Build the classifier prompt from config (prompt_file under CLASSIFIER)
# classifier_prompt, _ = make_chat_prompt_for_route("CLASSIFIER")


def node__classify_user_intent(state: ChatState) -> ChatState:
    # Use the last user message to classify the route and confidence
    last_message = state["messages"][-1].content

    # Create an LLM call that produces structured output matching UserIntentClassifier_output_format
    classifier_llm = llm.with_structured_output(
        UserIntentClassifier_output_format)

    # Build common template variables expected by route prompts and shared texts.
    # Some shared prompts reference `{from}` and route prompts may use `{text}` and `{history}`.
    # exclude latest user message from history
    prior_messages = state.get("messages", [])[:-1]
    history = _format_history(prior_messages)

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
        "history": history,
        "from": from_val,
        "routing_attempts": attempts,
        "triage_summary": triage_summary,
    }

    result = classifier_llm.invoke(
        classifier_prompt.format_messages(**fmt_kwargs))

    print(
        f"---> Inside: node__classify_user_intent .....Determined handling channel: {result.handling_channel}, (confidence: {result.confidence})\n"
    )

    # If confidence is high enough OR attempts exceeded, lock route and proceed
    if float(result.confidence) >= ROUTE_LOCK_THRESHOLD or attempts >= MAX_ROUTING_ATTEMPTS:
        return {
            "handling_channel": result.handling_channel,
            "confidence": float(result.confidence),
            "locked_route": result.handling_channel,
            "triage_question": None,
            "next": "route_by_user_intent",
        }

    # Low confidence: ask one clarifying question, increment attempts, and end the run (wait for user reply)
    question = (result.clarifying_question or "").strip(
    ) or _fallback_clarifying_question(result.handling_channel)
    return {
        "handling_channel": result.handling_channel,
        "confidence": float(result.confidence),
        "routing_attempts": attempts + 1,
        "triage_question": question,
        "messages": [AIMessage(content=question)],
        "next": "closed",
    }
