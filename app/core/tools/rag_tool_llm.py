from typing import Callable

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


class RagRetrievalArgs(BaseModel):
    """Arguments the LLM can pass to the RAG retrieval tool."""

    query: str = Field(
        description="User question to search in product manuals and documentation"
    )

    k: int = Field(
        default=3,
        description="Number of documents to retrieve",
        ge=1,
        le=10,
    )


def create_rag_retrieval_tool(retriever_fn: Callable, route_id: str) -> StructuredTool:
    """StructuredTool for RAG retrieval, bound to a specific route via closure.

    Args:
        retriever_fn: Callable(route_id, k) -> retriever (e.g. _get_route_retriever)
        route_id: The locked route identifier (e.g. "TPMS", "AA")
    """

    def _retrieve(query: str, k: int = 3) -> str:
        retriever = retriever_fn(route_id, k=k)
        docs = retriever.invoke(query)
        return "\n\n---\n\n".join(d.page_content for d in docs) if docs else ""

    return StructuredTool.from_function(
        name="rag_retrieval",
        func=_retrieve,
        args_schema=RagRetrievalArgs,
        description="""
Search product manuals and documentation.

Use this tool when the user asks about:

• troubleshooting or error symptoms (e.g. "no anda", "no funciona", "won't connect", "hace ruido", "making noise")
• setup or installation steps
• how-to guides or calibration procedures
• reset or configuration instructions
• technical specifications from the product manual

The tool returns relevant excerpts from the product documentation.
""",
    )
