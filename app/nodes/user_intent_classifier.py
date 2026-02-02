from typing import Literal
from pydantic import BaseModel, Field
from app.types import ChatState
from app.utils import make_chat_prompt_for_route, init_llm
from langchain_core.prompts import ChatPromptTemplate  # delete later


# Initialize the chat model used by the application
llm = init_llm(model="gpt-4o-mini", temperature=0)


class UserIntentClassifier_output_format(BaseModel):
    # Structured output model used to enforce the classifier's response shape
    handling_channel: Literal["TPMS", "AA", "CLIMATIZADOR"] = Field(
        ...,
        description="Clasifica el tipo de mensaje como 'TPMS', 'AA' o 'CLIMATIZADOR'.",
    )
    confidence: float


classifier_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a router for a WhatsApp bot. Classify the user's message .\n"
     "- handling_channel: 'sales' if the user asks about price, availability, buying, shipping, discounts, compatibility to purchase.\n"
     "- handling_channel: 'support' if the user reports a problem, installation issue, error, warranty, troubleshooting.\n"
     "Also choose product_family from the allowed list. If uncertain, pick the closest and lower confidence."),
    ("human", "{message}")
])
# classifier_prompt, _ = make_chat_prompt_for_route(
#     "CLASSIFIER", "User: {user_text}.")


def node__classify_user_intent(state: ChatState) -> ChatState:

    # Use the last user message to classify whether its intent requires a sales or support request
    last_message = state["messages"][-1].content

    # Create an LLM call that produces structured output matching UserIntentClassifier_output_format
    classifier_llm = llm.with_structured_output(
        UserIntentClassifier_output_format)

    print("Invoking classify_user_intent_node...")

    # # Build common template variables expected by route prompts and shared texts.
    # # Some shared prompts reference `{from}` and route prompts may use `{text}` and `{history}`.
    # history = "\n".join(m.content for m in state.get("messages", [])[:-1])
    # # Try to extract sender/phone info from the last message metadata if present
    # last_msg_obj = state["messages"][-1]
    # from_val = ""
    # try:
    #     meta = getattr(last_msg_obj, "metadata", None)
    #     if isinstance(meta, dict):
    #         from_val = meta.get("from", "") or meta.get("sender", "")
    # except Exception:
    #     from_val = ""

    # fmt_kwargs = {"user_text": last_message,
    #               "text": last_message, "history": history, "from": from_val}

    # result = classifier_llm.invoke(
    #     classifier_prompt.format_messages(**fmt_kwargs))

    result = classifier_llm.invoke(
        classifier_prompt.format_messages(message=last_message))

    print(
        f"Determined handling channel: {result.handling_channel},  (confidence: {result.confidence})")

    return {
        "handling_channel": result.handling_channel,
        "confidence": result.confidence,
    }
