from langgraph.graph import StateGraph, START, END
from app.types import ChatState
# Use route-based helper that builds the prompt from route_id
from app.prompts.prompt_utils import make_chat_prompt_for_route
from app.utils import init_llm
# from app.nodes.rag import get_retriever
# from app.tools.catalog_tool import catalog_lookup


def make_route_subgraph(route_id: str) -> StateGraph:
    """Construct a StateGraph subgraph for a given route.

    This builds and compiles a small retrieval-augmented generation (RAG)
    pipeline using a LangGraph `StateGraph`. The graph nodes perform the
    following steps:
    - `retrieve`: call the vector store retriever to obtain context documents
    - `generate`: call the LLM to produce an answer using the prompt + context
    - `enforce_limits`: shorten the answer if it exceeds `max_chars`
    - `maybe_handoff`: optionally append a WhatsApp handoff link after N attempts

    Parameters:
    - route_id (str): Key used to lookup route configuration in
        `config/routes.yaml` and to locate route-specific indexes/prompts.

    Returns:
    - A compiled `StateGraph` ready to be invoked by the application.
    """

    # Build prompt and get route config by passing only the route id
    subgraph_prompt, route_cfg = make_chat_prompt_for_route(
        route_id,
        "User: {user_text}\n\nRetrieved context:\n\n\nIf you need price/link, ask to call catalog_lookup.",

        # "User: {user_text}\n\nRetrieved context:\n{context}\n\nIf you need price/link, ask to call catalog_lookup.",
    )

    llm = init_llm(model="gpt-4o-mini", temperature=0.2)

    handoff_after = route_cfg.get("handoff_after_attempts")

    # def retrieve(state: ChatState) -> ChatState:
    #     """Retrieve relevant documents for the latest user message.

    #     Uses the route-specific retriever to fetch documents and stores a
    #     lightweight representation in `state['retrieved']`.
    #     """
    #     user_text = state["messages"][-1].content
    #     retriever = get_retriever(route_id, k=5)
    #     docs = retriever.invoke(user_text)
    #     state["retrieved"] = [
    #         {"page_content": d.page_content, "metadata": d.metadata} for d in docs
    #     ]
    #     return state

    def generate(state: ChatState) -> ChatState:
        """Generate an answer using the LLM and retrieved context.

        If the user appears to ask for price/SKU/link, call the catalog tool
        and append its output to the context before invoking the LLM.
        """
        user_text = state["messages"][-1].content
        print(f"Invoking node for handling route: {route_id}...")

        reply = llm.invoke(
            subgraph_prompt.format_messages(user_text=user_text))

        # Return the assistant reply wrapped in the state's messages field
        return {"messages": [reply]}

        # context = "\n\n".join(d["page_content"]
        #                       for d in (state.get("retrieved") or []))

        # if any(x in user_text.lower() for x in ["precio", "vale", "cuesta", "link", "comprar", "sku"]):
        #     tool_out = catalog_lookup(
        #         user_text, product_family=route_id, k=3)
        #     context += "\n\nCATALOG_LOOKUP:\n" + str(tool_out)

        # msg = llm.invoke(prompt.format_messages(
        #     user_text=user_text, context=context))
        # state["answer"] = msg.content
        # return state

    # def enforce_limits(state: ChatState) -> ChatState:
    #     """Ensure the generated answer respects the configured character limit.

    #     If the answer is too long, compress it using a smaller model.
    #     """
    #     ans = state.get("answer") or ""
    #     if len(ans) <= max_chars:
    #         return state

    #     compressor = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #     compress_prompt = ChatPromptTemplate.from_messages([
    #         ("system",
    #          f"Shorten to <= {max_chars} characters, keep links intact, keep meaning."),
    #         ("human", "{text}")
    #     ])
    #     state["answer"] = compressor.invoke(
    #         compress_prompt.format_messages(text=ans)).content
    #     return state

    # def maybe_handoff(state: ChatState) -> ChatState:
    #     """Optionally append a WhatsApp handoff after repeated attempts."""
    #     if not handoff_after:
    #         return state
    #     attempts = state["attempts"].get(route_id, 0)
    #     state["attempts"][route_id] = attempts + 1
    #     if state["attempts"][route_id] >= handoff_after:
    #         tech = cfg[route_id]["whatsapp"]["tech"]
    #         state["answer"] += f"\n\nSi querés, te derivamos por WhatsApp: {tech}"
    #     return state

    g = StateGraph(ChatState)
    # g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    # g.add_node("enforce_limits", enforce_limits)
    # g.add_node("maybe_handoff", maybe_handoff)
    g.add_edge(START, "generate")  # delete later
    g.add_edge("generate", END)  # delete later

    # g.add_edge(START, "retrieve")
    # g.add_edge("retrieve", "generate")
    # g.add_edge("generate", "enforce_limits")
    # g.add_edge("enforce_limits", "maybe_handoff")
    # g.add_edge("maybe_handoff", END)
    return g.compile()
