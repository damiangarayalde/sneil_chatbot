from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from app.core.graph.state import ChatState, get_history_and_last_msg, get_last_msg
from app.core.graph.msg_heuristics_no_llm import should_retrieve
from app.core.prompts.builders import make_chat_prompt_for_route
from app.core.utils import init_llm
from app.core.tools.rag import get_retriever
# from app.tools.catalog_tool import catalog_lookup
import time

# Structured output schema for the handler


class HandlerOutput(BaseModel):
    """Decide if the topic is still relevant or if a switch is needed."""
    is_topic_switch: bool = Field(
        description="True if the user changed the topic to a different product.")
    answer: str = Field(
        description="The response to the user. Empty if is_topic_switch is True.")


def make_route_subgraph(route_id: str) -> StateGraph:
    """Construct a StateGraph subgraph for a given route.

    Nodes:
    - retrieve: Fetches context documents from vector store based on user input.
    - generate: Uses context and history to produce a structured answer.
    - enforce_limits: shorten the answer if it exceeds `max_chars`
    - maybe_handoff: optionally append a WhatsApp handoff link after N attempts

    Parameters:
    - route_id (str): Key used to lookup route configuration in
        `config/routes.yaml` and to locate route-specific indexes/prompts.

    Returns:
    - A compiled `StateGraph` ready to be invoked by the application.
    """

    # Initialize LLM and Prompt
    llm = init_llm(model="gpt-4o-mini", temperature=0)
    prompt_template, route_cfg = make_chat_prompt_for_route(route_id)
    chain = prompt_template | llm.with_structured_output(HandlerOutput)

    # Max number of iterative solution attempts before we suggest human handoff.
    # (Config key: max_solving_attempts; supports legacy 'handoff_after_attempts'.)
    max_solving_attempts = int(
        route_cfg.get("max_solving_attempts")
        or route_cfg.get("handoff_after_attempts")
        or 0
    )

    def route_to_retrieve_or_generate(state: ChatState) -> str:
        last_msg = get_last_msg(state.get("messages") or [])
        return "retrieve" if should_retrieve(last_msg) else "generate"

    def retrieve(state: ChatState) -> ChatState:
        """
        Retrieves context documents for the specific product/route based on user query.
        Then and stores them as dictionaries in the state.    
        """
        last_msg = state["messages"][-1].content if state.get(
            "messages") else ""

        # 2. Initialize the retriever for this specific route (e.g., 'TPMS', 'AA')
        retriever = get_retriever(route_id, k=5)
        retrieved_docs = retriever.invoke(last_msg)
        print(f"Retrieved: {len(retrieved_docs)}")

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
        context_text = "\n\n".join(d.get("page_content", "") for d in docs)

        response = chain.invoke({
            "history": history,         # chat history
            "user_text": last_msg,      # current user msg
            "context": context_text,    # optional; rag_data
            "meta": ""
            # any route extras go here
        })

        # If the user switched product/topic, clear lock so the hub can re-route.
        if response.is_topic_switch:
            # Clear lock and signal triage
            return {
                "locked_route": None,
                "retrieved": None,
                "confidence": 0,
            }
        dt_ms = (time.perf_counter() - t0) * 1000

        print(f"Elapsed: {dt_ms:.1f} ms")
        # Count one "solving attempt" each time we send an LLM answer back.
        attempts_so_far = state.get("attempts") or 0 + 1

        answer_text = response.answer
        escalated = bool(state.get("escalated_to_human", False))
        if max_solving_attempts and attempts_so_far >= max_solving_attempts:
            escalated = True
            # Append handoff link only the first time we reach the threshold.
            if attempts_so_far == max_solving_attempts:
                tech = (route_cfg.get("whatsapp") or {}).get("tech")
                if tech:
                    answer_text += f"\n\nDisculpame, le puse garra pero parece que me falta informacion o no puedo darte esa respuesta. Si querés, te derivamos por WhatsApp: {tech}"

        # 4. Return the new assistant message (and any other state updates)
        return {
            "messages": [AIMessage(content=answer_text)],
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

    def maybe_handoff(state: ChatState) -> ChatState:
        # """Optionally append a WhatsApp handoff after repeated attempts."""
        # if not handoff_after:
        #     return state
        # attempts = state["attempts"].get(route_id, 0)
        # state["attempts"][route_id] = attempts + 1
        # if state["attempts"][route_id] >= handoff_after:
        #     tech = cfg[route_id]["whatsapp"]["tech"]
        #     state["answer"] += f"\n\nSi querés, te derivamos por WhatsApp: {tech}"
        return state

    # --- Graph Construction ---
    g = StateGraph(ChatState)

    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    # g.add_node("enforce_limits", enforce_limits)
    # g.add_node("maybe_handoff", maybe_handoff)

    # Flow: Start -> Retrieve Context -> Generate Answer -> End
    g.add_conditional_edges(START, route_to_retrieve_or_generate, {
        "retrieve": "retrieve",
        "generate": "generate",
    })
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    # g.add_edge("generate", "enforce_limits")
    # g.add_edge("enforce_limits", "maybe_handoff")
    # g.add_edge("maybe_handoff", END)
    return g.compile()
