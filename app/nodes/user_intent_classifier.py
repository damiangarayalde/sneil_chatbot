from pydantic import BaseModel
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

ROUTES = ["TPMS", "AA", "CLIMATIZADOR", "GENKI",
          "CARJACK", "MAYORISTA", "CALDERA", "UNKNOWN"]


class UserIntentClassifier_output_format(BaseModel):
    handling_channel: Literal["sales", "support"]
    product_family: Literal["TPMS", "AA", "CLIMATIZADOR", "GENKI",
                            "CARJACK", "MAYORISTA", "CALDERA", "UNKNOWN"]
    confidence: float


router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

router_prompt = ChatPromptTemplate.from_messages([
    ("system",  "Return only JSON with the handling channel, product family, and confidence. Do not include any other text."),
    ("user",
     "User message: {text}\nReturn handling channel, product family, and confidence. Allowed product families: " + ", ".join(ROUTES))
])


def node__classify_user_intent(state):
    text = state["messages"][-1].content
    out = router_llm.with_structured_output(UserIntentClassifier_output_format).invoke(
        router_prompt.format_messages(text=text))
    state["product_family"] = out.product_family
    state["handling_channel"] = out.handling_channel
    state["confidence"] = out.confidence
    print(
        f"Determined handling channel: {out.handling_channel}, product family: {out.product_family} (confidence: {out.confidence})")
    return state
