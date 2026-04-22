import json
import re
from pathlib import Path
from typing import Optional, Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


CATALOG_PATH = _repo_root() / "data" / "catalog.json"

# Maps CSV/user-facing family names → canonical catalog family names (all uppercase)
FAMILY_ALIASES: Dict[str, str] = {
    "caldera": "CALDERA",
    "aa": "AA",
    "aire acondicionado": "AA",
    "tpms": "TPMS",
    "climatizador": "CLIMATIZADOR",
    "carjack": "CARJACK",
    "arrancador": "CARJACK",
    "inflador": "CARJACK",
    "genki": "GENKI",
    "estacion de carga": "GENKI",
    "bluetti": "GENKI",
    "solar": "SOLAR",
    "panel solar": "SOLAR",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _resolve_family(family: str) -> str:
    """Resolve a user-supplied family name to its canonical catalog value."""
    key = _norm(family)
    return FAMILY_ALIASES.get(key, family.upper())


def catalog_lookup(query: str, product_family: Optional[str] = None, k: int = 3) -> Dict:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    q = _norm(query)

    resolved_family = _resolve_family(
        product_family) if product_family else None

    candidates = []
    for p in data["items"]:
        if resolved_family and p.get("product_family") != resolved_family:
            continue

        hay = " ".join(filter(None, [
            p.get("title", ""),
            p.get("sku", ""),
            p.get("model", ""),
            p.get("product_family", ""),
        ]))
        hay_n = _norm(hay)

        score = 0
        if q in hay_n:
            score += 5
        score += sum(1 for t in q.split() if t in hay_n)

        if score > 0:
            candidates.append((score, p))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in candidates[:k]]

    return {"matches": top, "count": len(top), "currency": data.get("defaults", {}).get("currency", "ARS")}
