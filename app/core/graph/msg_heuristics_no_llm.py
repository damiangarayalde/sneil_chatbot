"""Text heuristics & lightweight intent helpers.

This module started as `small_talk_filter.py` and now also hosts:
- low-info / small-talk detection (avoid locking route)
- human-handoff detection
- cheap keyword-based direct routing
- default clarifier + route disambiguation questions

"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Set

from app.core.utils import (
    get_route_clarifying_question,
    get_route_mentions,
    get_routes,
    load_cfg,
)

# --------------------------------------------------------------------------------------
# Low-info / small talk

LOW_INFO_MSGS = {
    "hola", "buenas", "buen día", "buen dia", "buenas tardes", "buenas noches"  # ,
    # "ok", "oka", "dale", "listo", "joya", "perfecto", "bien",
    #  "si", "sí", "no", "seguro", "claro", "gracias", "👍", "👌", "🙏"
}


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def is_low_info(text: str) -> bool:
    t = normalize(text)
    if not t:
        return True
    if len(t) <= 3:
        return True
    if t in LOW_INFO_MSGS:
        return True
    return False


# --------------------------------------------------------------------------------------
# Retrieval heuristic (kept here because some routes use it)

TROUBLE_KEYWORDS = [
    # troubleshooting / failure
    "no funciona", "no anda", "no prende", "no enciende", "no carga", "no conecta",
    "no enfría", "no enfria", "no calienta", "no responde", "se corta", "se apaga",
    "error", "falla", "problema", "ruido", "vibra", "pierde", "pierde presión", "pierde presion",
    "alarma", "sensor", "emparejar", "bluetooth", "presión", "presion", "temperatura",
    # how-to / setup
    "cómo", "como", "instalar", "configurar", "calibrar", "reset", "reiniciar",
    "paso a paso", "manual", "instrucciones", "setear", "programar",
]


def should_retrieve(user_text: str) -> bool:
    t = normalize(user_text)
    if is_low_info(t):
        return False
    # retrieval when it looks like a question or an issue/how-to
    return ("?" in t) or any(k in t for k in TROUBLE_KEYWORDS)


# --------------------------------------------------------------------------------------
# Human handoff detection

_HUMAN_REQUEST_PATTERNS = [
    r"hablar con (un|una) (humano|persona|asesor|agente|operador)",
    r"(asesor|agente|operador) (humano|real)",
    r"(quiero|necesito) (un|una) (humano|persona|asesor|agente|operador)",
    r"humano",
    r"representante",
    r"llamar( me)?",
    r"whats(app)?",
    r"wpp",
    r"wapp",
]
_HUMAN_REQUEST_RE = re.compile(
    "|".join(f"(?:{p})" for p in _HUMAN_REQUEST_PATTERNS), re.IGNORECASE)


def asked_for_human(text: str) -> bool:
    return bool(_HUMAN_REQUEST_RE.search(text or ""))


def escalation_message() -> str:
    """Message returned when we decide to hand off to a human.

    You can set HUMAN_WAPP_NUMBER env var for a concrete number.
    """
    num = os.getenv("HUMAN_WAPP_NUMBER", "").strip()
    if num:
        return f"Para atención humana, escribinos por WhatsApp a {num}."
    return "Para atención humana, escribinos por WhatsApp a nuestro número de soporte (humano)."


# --------------------------------------------------------------------------------------
# Cheap keyword routing (support/sale + single route mention)

_SUPPORT_SALE_WORDS = [
    # Spanish
    "soporte", "ayuda", "problema", "error", "falla", "reclamo",
    "venta", "comprar", "precio", "presupuesto", "cotizacion", "cotización", "stock",
    # English
    "support", "sale", "buy", "price", "quote",
]

# Curated route mention aliases (used ONLY for the fast path)
_ROUTE_MENTIONS: Dict[str, List[str]] = {
    "TPMS": ["tpms", "sensor tpms", "sensores tpms", "sensor", "sensores"],
    "AA": ["aa", "a/a", "aire acondicionado", "aire", "ac"],
    "CLIMATIZADOR": ["climatizador", "climatización", "caldera", "air heater"],
}


def contains_support_or_sale(text: str) -> bool:
    t = normalize(text)
    return any(w in t for w in _SUPPORT_SALE_WORDS)


def reload_route_heuristics_cache() -> None:
    """Clear the in-process cache for route mentions (next call re-reads config)."""
    _route_mentions_map_from_cfg.cache_clear()


@lru_cache(maxsize=1)
def _route_mentions_map_from_cfg() -> Dict[str, List[str]]:
    """Build the route->mentions map from config (cached)."""
    cfg = load_cfg()
    out: Dict[str, List[str]] = {}
    for route_id in get_routes(cfg):
        out[route_id] = get_route_mentions(route_id, cfg)
    return out


def direct_route_from_keywords(text: str, allowed_routes: Set[str]) -> Optional[str]:
    """Fast-path routing.

    If the message clearly indicates 'support/sale' AND mentions exactly one route,
    we can lock without spending an LLM call.

    Route aliases/synonyms come from config (no hardcoded product lists here).
    """
    t = normalize(text)
    if not t or not contains_support_or_sale(t):
        return None

    mentions_map = _route_mentions_map_from_cfg()

    hits: List[str] = []
    for route in allowed_routes:
        route_hits = [route.lower()]
        route_hits += mentions_map.get(route, [])
        if any(k in t for k in route_hits):
            hits.append(route)

    # stable unique
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return hits[0]
    return None


# --------------------------------------------------------------------------------------
# Default clarifiers / disambiguation messages

def default_clarifier() -> str:
    return "¡Hola! ¿En qué puedo ayudarte hoy?"


def wrap_with_greeting(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return default_clarifier()
    # Avoid double greetings
    if normalize(t).startswith("hola"):
        return t
    return f"¡Hola! {t}"


def route_disambiguation_question(route_guess: Optional[str]) -> str:
    """Route-specific fallback if you already have a good guess.

    The per-route question (if any) is read from config under:
      <ROUTE_ID>.heuristics.clarifying_question

    If missing, we fall back to a generic question (still no hardcoded route logic).
    """
    if route_guess:
        cfg = load_cfg()
        q = get_route_clarifying_question(route_guess, cfg)
        if q:
            return q

        # Generic fallback (route_guess might not exist in config)
        rg = (route_guess or "").strip()
        if rg:
            return (
                f"¿Tu consulta es sobre {rg}? "
                "¿Es por instalación/configuración o por un problema de funcionamiento?"
            )

    return "¿Podés decirme qué producto es y cuál es el problema principal?"
