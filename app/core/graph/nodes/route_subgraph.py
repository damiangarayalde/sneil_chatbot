from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from app.core.graph.state import ChatState, get_history_and_last_msg, get_last_msg
from app.core.graph.msg_heuristics_no_llm import (
    asked_for_human,
    escalation_message,
    route_disambiguation_question,
    should_retrieve,
    wrap_with_greeting,
)
from app.core.prompts.builders import make_chat_prompt_for_route
from app.core.utils import init_llm
from app.core.tools.rag import get_retriever
# from app.tools.catalog_tool import catalog_lookup

# -----------------------------------------------------------------------------
# Structured output schema for the route handler


class HandlerOutput(BaseModel):
    """Route handler output.

    - If the user switched product/topic, we want to clear lock so hub can re-route.
    - Otherwise we send back an answer.
    """
    is_topic_switch: bool = Field(
        description="True if the user changed the topic to a different product."
    )
    answer: str = Field(
        description="The response to the user. Empty if is_topic_switch is True."
    )

# -----------------------------------------------------------------------------
# Graph factory


def make_route_subgraph(route_id: str) -> StateGraph:
    """Construct a StateGraph subgraph for a given locked route.

    Intended behavior:
    1) START checks max solve attempts (and explicit human request). If exceeded => handoff + reset attempts.
    2) If attempts == 0 and message is vague => clarifying question (NO RAG, NO attempts increment)
    3) Otherwise => retrieve (RAG) => generate answer + separate confirmation => attempts += 1
    """

    # Initialize LLM and Prompt
    llm = init_llm(model="gpt-4o-mini", temperature=0)
    prompt_template, route_cfg = make_chat_prompt_for_route(route_id)
    chain = prompt_template | llm.with_structured_output(HandlerOutput)

    # Max number of iterative solution attempts before we suggest human handoff.
    max_attempts_before_handoff = int(
        route_cfg.get("max_attempts_before_handoff")
        or 0
    )

    # Confirmation message MUST be separate from the LLM answer (per spec).
    CONFIRMATION_MSG = "¿Esto te sirvió? Respondé Sí o No."

    def route_from_start(state: ChatState) -> Literal["handoff", "clarify", "retrieve"]:
        """Single decision point at START."""
        last_msg = get_last_msg(state.get("messages") or [])
        attempts = int(state.get("attempts") or 0)

        # 0) explicit human request always wins
        if asked_for_human(last_msg):
            return "handoff"

        # 1) max attempts gate (check BEFORE any solving work)
        if max_attempts_before_handoff and attempts >= max_attempts_before_handoff:
            return "handoff"

        # 2) first attempt and vague message => clarifying question (no RAG, no attempts increment)
        if attempts == 0 and not should_retrieve(last_msg):
            return "clarify"

        # 3) otherwise, solve path (always includes retrieval)
        return "retrieve"

    def handoff(state: ChatState) -> ChatState:
        """Apologize + recommend human support. Reset attempts."""
        msg = (
            "Disculpá — intenté ayudarte, pero me falta información o no puedo confirmar una solución con lo que tengo.\n\n"
            f"{escalation_message()}"
        )
        return {
            "messages": [AIMessage(content=msg)],
            "attempts": 0,
            "escalated_to_human": True,
            "locked_route": None,
            "confidence": 0,
            "retrieved": None,
        }

    def clarify(state: ChatState) -> ChatState:
        """Ask for clarification using heuristics (no LLM, no RAG)."""
        q = route_disambiguation_question(route_id)
        return {
            "messages": [AIMessage(content=wrap_with_greeting(q))],
            "retrieved": None,
        }

    def retrieve(state: ChatState) -> ChatState:
        """Retrieve context documents for the route based on the user query."""
        last_msg = get_last_msg(state.get("messages") or [])
        retriever = get_retriever(route_id, k=3)
        retrieved_docs = retriever.invoke(last_msg)
        print(f"[{route_id}] Retrieved docs: {len(retrieved_docs)}")
        return {"retrieved": [d.dict() for d in retrieved_docs]}

    def generate(state: ChatState) -> ChatState:
        """Generate an answer using the LLM and retrieved context.

        If the user appears to ask for price/SKU/link, call the catalog tool
        and append its output to the context before invoking the LLM.
        """
        t0 = time.perf_counter()

        history, last_msg = get_history_and_last_msg(
            state.get("messages") or [])

        # if any(x in user_text.lower() for x in ["precio", "vale", "cuesta", "link", "comprar", "sku"]):
        #     tool_out = catalog_lookup(
        #         user_text, product_family=route_id, k=3)
        #     context += "\n\nCATALOG_LOOKUP:\n" + str(tool_out)

        # Prepare context string from retrieved docs
        docs = state.get("retrieved") or []
        context_text = "\n\n".join((d.get("page_content") or "") for d in docs)

        response = chain.invoke({
            "history": history,         # chat history
            "user_text": last_msg,      # current user msg
            "context": context_text,    # optional; rag_data
            "meta": ""
            # any route extras go here
        })

        # If the user switched product/topic, clear lock so the hub can re-route.
        if response.is_topic_switch:
            return {
                "locked_route": None,
                "retrieved": None,
                "confidence": 0,
                "attempts": 0,
            }

        dt_ms = (time.perf_counter() - t0) * 1000

        print(f"[{route_id}] LLM elapsed: {dt_ms:.1f} ms")

        # Count one "solving attempt" each time we send an LLM answer back.
        attempts_so_far = int(state.get("attempts") or 0) + 1

        answer_text = (response.answer or "").strip()
        if not answer_text:
            answer_text = "Entiendo. ¿Podés darme un poco más de detalle para ayudarte mejor?"

        escalated = bool(state.get("escalated_to_human", False))

        return {
            "messages": [
                AIMessage(content=answer_text),
                AIMessage(content=CONFIRMATION_MSG),
            ],
            "attempts": attempts_so_far,
            "escalated_to_human": escalated,
        }

    def enforce_limits(state: ChatState) -> ChatState:
        # """Ensure the generated answer respects the configured character limit.

        # If the answer is too long, compress it using a smaller model.
        # """
        # ans = state.get("answer") or ""
        # if len(ans) <= max_chars:
        #     return state

        # compressor = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # compress_prompt = ChatPromptTemplate.from_messages([
        #     ("system",
        #      f"Shorten to <= {max_chars} characters, keep links intact, keep meaning."),
        #     ("human", "{text}")
        # ])
        # state["answer"] = compressor.invoke(
        #     compress_prompt.format_messages(text=ans)).content
        return state

    # --- Graph Construction ---
    g = StateGraph(ChatState)

    g.add_node("handoff", handoff)
    g.add_node("clarify", clarify)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    # g.add_node("enforce_limits", enforce_limits)

    # Start decision (gate + clarify vs solve)
    g.add_conditional_edges(
        START,
        route_from_start,
        {
            "handoff": "handoff",
            "clarify": "clarify",
            "retrieve": "retrieve",
        },
    )

    g.add_edge("handoff", END)
    g.add_edge("clarify", END)
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    # g.add_edge("generate", "enforce_limits")
    # g.add_edge("enforce_limits", "maybe_handoff")
    return g.compile()
