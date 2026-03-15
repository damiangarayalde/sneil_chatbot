from typing import Optional, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from app.core.tools.catalog_tool import catalog_lookup


class CatalogLookupArgs(BaseModel):
    """Arguments the LLM can pass to the catalog lookup tool."""

    query: str = Field(
        description="User question about a product. Example: 'precio del C260'"
    )

    product_family: Optional[str] = Field(
        default=None,
        description="Optional family filter like TPMS, AA, CLIMATIZADOR, CARJACK",
    )

    k: int = Field(
        default=3,
        description="Maximum number of results to return",
        ge=1,
        le=10,
    )


def _catalog_lookup_tool(
    query: str,
    product_family: Optional[str] = None,
    k: int = 3,
) -> Dict[str, Any]:
    """Wrapper that calls the internal catalog lookup."""
    return catalog_lookup(query=query, product_family=product_family, k=k)


def create_catalog_lookup_tool() -> StructuredTool:
    """
    Reusable tool for querying the Neil product catalog.
    """
    return StructuredTool.from_function(
        name="catalog_lookup",
        func=_catalog_lookup_tool,
        args_schema=CatalogLookupArgs,
        description="""
Search Neil's internal product catalog.

Use this tool when the user asks about:

• product prices
• product links
• SKUs or product codes
• product availability
• product specs or images

The tool returns structured product matches including:
- id / SKU
- title
- price
- sale_price
- link
- image_link
- family
- category
- availability

Use the returned information to answer the user.
""",
    )
