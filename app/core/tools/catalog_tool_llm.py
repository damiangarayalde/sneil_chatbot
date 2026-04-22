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
        description=(
            "Optional family filter. Accepted values (case-insensitive): "
            "AA, Aire Acondicionado, "
            "Caldera, "
            "TPMS, "
            "Climatizador, "
            "Carjack, Arrancador, Inflador, "
            "Genki, Estacion de carga, Bluetti, "
            "Solar, Panel solar"
        ),
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
        description=(
            "Search Neil's internal product catalog. "
            "Use when the user asks about prices, links, SKUs, availability, or specs. "
            "Product families: AA (air conditioners), Caldera (diesel heaters), "
            "TPMS (tire pressure sensors), Climatizador (evaporative coolers), "
            "Carjack/Arrancador/Inflador (jump starters, jacks, compressors), "
            "Genki/Estacion de carga/Bluetti (power stations), "
            "Solar/Panel solar (solar panels and kits). "
            "Returns matches with fields: product_family, model, sku, title, price (ARS), sales_link. "
            "Present each match as: '**<title>**\\n- Precio: $<price> ARS\\n- Link de compra: <sales_link>'. "
            "Only include fields that are present; never invent prices or links."
        ),
    )
