import yaml
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.nodes.rag import get_retriever
from app.tools.catalog_tool import catalog_lookup
from app.types import ChatState


cfg = yaml.safe_load(Path("config/products.yaml").read_text(encoding="utf-8"))


def load_prompt(path: str) -> str:
    """Load a prompt template file from disk.

    Parameters:
    - path (str): Relative path to the prompt file.

    Returns:
    - str: The file contents as a UTF-8 decoded string.
    """
    return Path(path).read_text(encoding="utf-8")


BASE = load_prompt("app/prompts/shared/base_policy.md")
WHATSAPP = load_prompt("app/prompts/shared/whatsapp_format.md")
ESCALATION = load_prompt("app/prompts/shared/escalation_policy.md")
SHIP_SPAIN = load_prompt("app/prompts/shared/shipping_spain.md")


def make_product_subgraph(product_id: str) -> StateGraph:
    """Construct a StateGraph subgraph for a given product.

    This builds and compiles a small retrieval-augmented generation (RAG)
    pipeline using a LangGraph `StateGraph`. The graph nodes perform the
    following steps:
    - `retrieve`: call the vector store retriever to obtain context documents
    - `generate`: call the LLM to produce an answer using the prompt + context
    - `enforce_limits`: shorten the answer if it exceeds `max_chars`
    - `maybe_handoff`: optionally append a WhatsApp handoff link after N attempts

    Parameters:
    - product_id (str): Key used to lookup product configuration in
        `config/products.yaml` and to locate product-specific indexes/prompts.

    Returns:
    - A compiled `StateGraph` ready to be invoked by the application.
    """

    # Load the product-specific prompt and configuration values
    product_prompt = load_prompt(cfg[product_id]["prompt_file"])
    max_chars = cfg[product_id]["max_chars"]
    handoff_after = cfg[product_id].get("handoff_after_attempts")

    llm = ChatOpenAI(model="gpt-5 mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         BASE + "\n\n" + WHATSAPP + "\n\n" + SHIP_SPAIN + "\n\n" + ESCALATION + "\n\n"
         + f"## PRODUCT: {product_id}\n" + product_prompt
         + f"\n\nHard limit: {max_chars} characters including spaces."
         ),
        ("user", "User: {user_text}\n\nRetrieved context:\n{context}\n\nIf you need price/link, ask to call catalog_lookup.")
    ])

    def retrieve(state: ChatState) -> ChatState:
        """Retrieve relevant documents for the latest user message.

        Uses the product-specific retriever to fetch documents and stores a
        lightweight representation in `state['retrieved']`.
        """
        user_text = state["messages"][-1].content
        retriever = get_retriever(product_id, k=5)
        docs = retriever.invoke(user_text)
        state["retrieved"] = [
            {"page_content": d.page_content, "metadata": d.metadata} for d in docs
        ]
        return state

    def generate(state: ChatState) -> ChatState:
        """Generate an answer using the LLM and retrieved context.

        If the user appears to ask for price/SKU/link, call the catalog tool
        and append its output to the context before invoking the LLM.
        """
        user_text = state["messages"][-1].content
        context = "\n\n".join(d["page_content"]
                              for d in (state.get("retrieved") or []))

        if any(x in user_text.lower() for x in ["precio", "vale", "cuesta", "link", "comprar", "sku"]):
            tool_out = catalog_lookup(
                user_text, product_family=product_id, k=3)
            context += "\n\nCATALOG_LOOKUP:\n" + str(tool_out)

        msg = llm.invoke(prompt.format_messages(
            user_text=user_text, context=context))
        state["answer"] = msg.content
        return state

    def enforce_limits(state: ChatState) -> ChatState:
        """Ensure the generated answer respects the configured character limit.

        If the answer is too long, compress it using a smaller model.
        """
        ans = state.get("answer") or ""
        if len(ans) <= max_chars:
            return state

        compressor = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        compress_prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"Shorten to <= {max_chars} characters, keep links intact, keep meaning."),
            ("user", "{text}")
        ])
        state["answer"] = compressor.invoke(
            compress_prompt.format_messages(text=ans)).content
        return state

    def maybe_handoff(state: ChatState) -> ChatState:
        """Optionally append a WhatsApp handoff after repeated attempts."""
        if not handoff_after:
            return state
        attempts = state["attempts"].get(product_id, 0)
        state["attempts"][product_id] = attempts + 1
        if state["attempts"][product_id] >= handoff_after:
            tech = cfg[product_id]["whatsapp"]["tech"]
            state["answer"] += f"\n\nSi querés, te derivamos por WhatsApp: {tech}"
        return state

    g = StateGraph(ChatState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("enforce_limits", enforce_limits)
    g.add_node("maybe_handoff", maybe_handoff)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "enforce_limits")
    g.add_edge("enforce_limits", "maybe_handoff")
    g.add_edge("maybe_handoff", END)
    return g.compile()
