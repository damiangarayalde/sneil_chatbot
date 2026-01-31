from pydantic import BaseModel
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

ROUTES = ["TPMS", "AA", "CLIMATIZADOR", "GENKI",
          "CARJACK", "MAYORISTA", "CALDERA", "UNKNOWN"]


class RouteOut(BaseModel):
    route: Literal["TPMS", "AA", "CLIMATIZADOR", "GENKI",
                   "CARJACK", "MAYORISTA", "CALDERA", "UNKNOWN"]
    confidence: float


router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

router_prompt = ChatPromptTemplate.from_messages([
    ("system",  "Return only JSON with the route and confidence. Do not include any other text."),
    ("user",
     "User message: {text}\nReturn route and confidence. Allowed routes: " + ", ".join(ROUTES))
])


def route_node(state):
    text = state["messages"][-1].content
    out = router_llm.with_structured_output(RouteOut).invoke(
        router_prompt.format_messages(text=text))
    state["route"] = out.route
    return state
