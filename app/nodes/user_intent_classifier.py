from typing import Literal, Optional
from pydantic import BaseModel, Field
from app.types import ChatState
from app.prompts.prompt_utils import make_chat_prompt_for_route
from app.utils import init_llm
from langchain_core.messages import AIMessage


# Initialize the chat model used by the application
llm = init_llm(model="gpt-4o-mini", temperature=0)

# --- Step 3A routing configuration (keep here for now; can move later) ---
ROUTE_LOCK_THRESHOLD = 0.75
MAX_ROUTING_ATTEMPTS = 3


class UserIntentClassifier_output_format(BaseModel):
    # Structured output model used to enforce the classifier's response shape

    handling_channel: Literal["TPMS", "AA", "CLIMATIZADOR"] = Field(
        ...,
        description="Clasifica el tipo de mensaje como 'TPMS', 'AA' o 'CLIMATIZADOR'.",
    )
    confidence: float = Field(..., ge=0, le=1)

    clarifying_question: Optional[str] = Field(
        None,
        description=(
            "If confidence is low, ask ONE short clarifying question that would most improve routing."
        ),
    )


classifier_prompt, _ = make_chat_prompt_for_route(
    "CLASSIFIER", "User: {user_text}."
)


def _fallback_clarifying_question(route_guess: str) -> str:
    # Fallback in case LLM does not return clarifying_question.
    if route_guess == "TPMS":
        return "¿Tu problema es sobre sensores TPMS que no aparecen/lecturas inestables, o sobre instalación/configuración?"
    if route_guess == "AA":
        return "¿Es sobre aire acondicionado (AA) instalación/potencia/consumo, o un problema de rendimiento (enfriamiento/calefacción)?"
    return "¿Es sobre requisitos de instalación del climatizador (techo/ventilación/potencia) o un problema de rendimiento (flujo de aire/enfriamiento)?"


def node__classify_user_intent(state: ChatState) -> ChatState:
    # Use the last user message to classify the route and confidence
    last_message = state["messages"][-1].content

    # Create an LLM call that produces structured output matching UserIntentClassifier_output_format
    classifier_llm = llm.with_structured_output(
        UserIntentClassifier_output_format)

    print("Invoking classify_user_intent_node...")

    # NOTE: Keep your commented formatting block. We'll reuse it in Step 3B.
    # Build common template variables expected by route prompts and shared texts.
    # Some shared prompts reference `{from}` and route prompts may use `{text}` and `{history}`.
    history = "\n".join(m.content for m in state.get("messages", [])[:-1])

    # Try to extract sender/phone info from the last message metadata if present
    last_msg_obj = state["messages"][-1]
    from_val = ""
    try:
        meta = getattr(last_msg_obj, "metadata", None)
        if isinstance(meta, dict):
            from_val = meta.get("from", "") or meta.get("sender", "")
    except Exception:
        from_val = ""

    fmt_kwargs = {"user_text": last_message,
                  "text": last_message, "history": history, "from": from_val}

    result = classifier_llm.invoke(
        classifier_prompt.format_messages(**fmt_kwargs))

    # result = classifier_llm.invoke(
    #     classifier_prompt.format_messages(message=last_message)
    # )

    print(
        f"Determined handling channel: {result.handling_channel}, (confidence: {result.confidence})")

    attempts = int(state.get("routing_attempts") or 0)

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
