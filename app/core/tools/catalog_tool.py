import json
import re
from pathlib import Path
from typing import Optional, List, Dict

# Define the path to the catalog JSON file
# CATALOG_PATH = Path("data/catalog.json")


def _repo_root() -> Path:
    # catalog_tool.py is expected at: <root>/app/core/tools/catalog_tool.py
    return Path(__file__).resolve().parents[3]


CATALOG_PATH = _repo_root() / "data" / "catalog.json"


def _norm(s: str) -> str:
    # Normalize the string by converting to lowercase and removing extra whitespace
    return re.sub(r"\s+", " ", s.lower().strip())


def catalog_lookup(query: str, product_family: Optional[str] = None, k: int = 3) -> Dict:
    """
    Searches for products in the catalog based on a query.

    Parameters:
    - query (str): The search term used to find matching products in the catalog.
    - product_family (Optional[str]): An optional parameter to filter results by a specific product family.
      If provided, only products belonging to this family will be considered.
    - k (int): An optional parameter that specifies the maximum number of matching products to return.
      Defaults to 3.

    Returns:
    - Dict: A dictionary containing:
        - matches: A list of the top matching products (up to `k` products).
        - count: The number of matches found.
        - currency: The currency used in the catalog, defaulting to "ARS" if not specified.
    """

    # Load the catalog data from the JSON file
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    q = _norm(query)  # Normalize the query string

    candidates = []  # List to hold matching products
    for p in data["items"]:
        # Skip products that do not match the specified product family
        if product_family and p.get("family") != product_family:
            continue
        score = 0  # Initialize score for the current product
        # Create a string containing all relevant fields for matching
        hay = " ".join([
            p.get("title", ""),
            p.get("description", ""),
            p.get("sku", ""),
            p.get("family", ""),
            p.get("google_product_category", "")
        ] + p.get("aliases", []))
        hay_n = _norm(hay)  # Normalize the product string
        if q in hay_n:
            score += 5  # Increase score for exact match
        # Light fuzzy matching: count shared tokens between query and product string
        score += sum(1 for t in q.split() if t in hay_n)
        if score > 0:
            # Add product to candidates if score is positive
            candidates.append((score, p))

    # Sort candidates by score in descending order
    candidates.sort(key=lambda x: x[0], reverse=True)
    # Get the top k matching products
    top = [p for _, p in candidates[:k]]

    # print(top)
    # Return the matches along with the count and currency
    return {"matches": top, "count": len(top), "currency": data.get("defaults", {}).get("currency", "ARS")}
