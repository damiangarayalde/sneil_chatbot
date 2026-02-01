from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Literal
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from app.types import ChatState


# ROUTES = ["TPMS", "AA", "CLIMATIZADOR", "GENKI",
#           "CARJACK", "MAYORISTA", "CALDERA", "UNKNOWN"]


load_dotenv()

# Initialize the chat model used by the application
llm = init_chat_model(model="gpt-4o-mini", temperature=0)


class UserIntentClassifier_output_format(BaseModel):
    # Structured output model used to enforce the classifier's response shape
    handling_channel: Literal["sales", "support"] = Field(
        ...,
        description="Clasifica el tipo de mensaje como 'ventas' o 'soporte'.",
    )
    # product_family: Literal["TPMS", "AA", "CLIMATIZADOR",
    #                         "GENKI", "CARJACK", "MAYORISTA", "CALDERA", "UNKNOWN"]
    confidence: float


classifier_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a router for a WhatsApp bot. Classify the user's message .\n"
     "- handling_channel: 'sales' if the user asks about price, availability, buying, shipping, discounts, compatibility to purchase.\n"
     "- handling_channel: 'support' if the user reports a problem, installation issue, error, warranty, troubleshooting.\n"
     "Also choose product_family from the allowed list. If uncertain, pick the closest and lower confidence."),
    ("human", "{message}")
])


def node__classify_user_intent(state: ChatState) -> ChatState:

    # Use the last user message to classify whether its intent requires a sales or support request
    last_message = state["messages"][-1].content

    # Create an LLM call that produces structured output matching UserIntentClassifier_output_format
    classifier_llm = llm.with_structured_output(
        UserIntentClassifier_output_format)

    print("Invoking classify_user_intent_node...")

    result = classifier_llm.invoke(
        classifier_prompt.format_messages(message=last_message))

    print(
        f"Determined handling channel: {result.handling_channel},  (confidence: {result.confidence})")
   # f"Determined handling channel: {result.handling_channel}, product family: {result.product_family} (confidence: {result.confidence})")

    return {
        "handling_channel": result.handling_channel,
        "confidence": result.confidence,
        # "product_family": result.product_family
    }
