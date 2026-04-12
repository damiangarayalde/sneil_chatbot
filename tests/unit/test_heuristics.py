"""
Unit tests for app/core/graph/msg_heuristics_no_llm.py.

All tests are pure Python — no LLM, no I/O, no config file required.
`direct_route_from_keywords` mocks the config-dependent mention map so
tests remain offline even when config.yaml is absent.

Run:
    pytest -m unit
    pytest tests/unit/test_heuristics.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from app.core.graph.msg_heuristics_no_llm import (
    is_low_info,
    asked_for_human,
    direct_route_from_keywords,
    contains_support_or_sale,
    normalize,
)

# ---------------------------------------------------------------------------
# Controlled mention map used to keep direct_route tests offline
# ---------------------------------------------------------------------------

_MOCK_MENTIONS: dict[str, list[str]] = {
    "TPMS": ["tpms", "sensor tpms", "sensores tpms", "sensor", "sensores"],
    "AA": ["aa", "a/a", "aire acondicionado", "aire", "ac"],
    "CLIMATIZADOR": ["climatizador", "climatización", "caldera"],
}


# ---------------------------------------------------------------------------
# is_low_info
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("text, expected", [
    # empty / whitespace
    ("", True),
    ("   ", True),
    # short (len ≤ 3 after strip)
    ("ok", True),
    ("hi", True),
    ("no", True),
    # exact LOW_INFO_MSGS matches (len > 3)
    ("hola", True),
    ("dale", True),
    ("gracias", True),
    ("buenas tardes", True),
    ("chau", True),
    ("listo", True),
    # meaningful text
    ("quiero comprar un sensor", False),
    ("no anda el tpms", False),
    ("tengo un problema con el aire acondicionado", False),
    ("¿cómo instalo el sensor?", False),
    ("precio del climatizador", False),
])
def test_is_low_info(text: str, expected: bool) -> None:
    assert is_low_info(text) is expected


# ---------------------------------------------------------------------------
# asked_for_human
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("text, expected", [
    # pattern: "hablar con (un|una) (humano|persona|asesor|agente|operador)"
    ("quiero hablar con una persona", True),
    ("hablar con un asesor por favor", True),
    # pattern: "(asesor|agente|operador) (humano|real)"
    ("necesito un asesor humano", True),
    ("agente real", True),
    # pattern: "(quiero|necesito) (un|una) (humano|persona|asesor|agente|operador)"
    ("quiero un humano", True),
    ("necesito una persona", True),
    ("quiero un operador", True),
    # pattern: "humano" (bare word)
    ("hay algún humano?", True),
    ("humano", True),
    # pattern: "representante"
    ("quiero hablar con un representante", True),
    # pattern: "llamar( me)?"
    ("puedo llamar?", True),
    ("llamarme", True),
    # pattern: "whats(app)?"
    ("whatsapp", True),
    ("whats", True),
    # pattern: "wpp"
    ("escribime por wpp", True),
    # pattern: "wapp"
    ("wapp", True),
    # negatives
    ("quiero saber el precio del tpms", False),
    ("", False),
    ("tengo un problema con el sensor", False),
    ("no funciona el aire", False),
])
def test_asked_for_human(text: str, expected: bool) -> None:
    assert asked_for_human(text) is expected


# ---------------------------------------------------------------------------
# contains_support_or_sale
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("text, expected", [
    ("soporte tpms", True),
    ("necesito ayuda", True),
    ("hay un problema", True),
    ("venta de sensores", True),
    ("precio del sensor", True),
    ("quiero comprar", True),
    ("stock disponible", True),
    # English words
    ("support needed", True),
    ("buy a sensor", True),
    # negatives
    ("hola", False),
    ("no sé", False),
    ("", False),
])
def test_contains_support_or_sale(text: str, expected: bool) -> None:
    assert contains_support_or_sale(text) is expected


# ---------------------------------------------------------------------------
# direct_route_from_keywords  (config mocked — fully offline)
# ---------------------------------------------------------------------------

def _run(text: str, allowed: set[str]) -> str | None:
    with patch(
        "app.core.graph.msg_heuristics_no_llm._route_mentions_map_from_cfg",
        return_value=_MOCK_MENTIONS,
    ):
        return direct_route_from_keywords(text, allowed)


@pytest.mark.unit
def test_direct_route_single_match_by_mention() -> None:
    assert _run("soporte tpms no conecta", {"TPMS", "AA"}) == "TPMS"


@pytest.mark.unit
def test_direct_route_single_match_by_route_name() -> None:
    # Route name itself (lowercased) is always checked
    assert _run("soporte AA", {"TPMS", "AA"}) == "AA"


@pytest.mark.unit
def test_direct_route_match_via_synonym() -> None:
    assert _run("venta aire acondicionado", {"TPMS", "AA"}) == "AA"


@pytest.mark.unit
def test_direct_route_ambiguous_returns_none() -> None:
    # Both TPMS and AA mentioned → cannot decide
    assert _run("soporte tpms y aire acondicionado", {"TPMS", "AA"}) is None


@pytest.mark.unit
def test_direct_route_no_support_sale_word_returns_none() -> None:
    # Route mentioned but no support/sale keyword
    assert _run("tpms", {"TPMS"}) is None


@pytest.mark.unit
def test_direct_route_empty_text_returns_none() -> None:
    assert _run("", {"TPMS", "AA"}) is None


@pytest.mark.unit
def test_direct_route_support_word_no_route_returns_none() -> None:
    # Support word present but no recognised route mentioned
    assert _run("soporte general", {"TPMS", "AA"}) is None


@pytest.mark.unit
def test_direct_route_case_insensitive() -> None:
    # Normalize lowercases before matching
    assert _run("Soporte TPMS", {"TPMS", "AA"}) == "TPMS"


@pytest.mark.unit
def test_direct_route_route_not_in_allowed_is_ignored() -> None:
    # CLIMATIZADOR not in allowed set — should not match
    assert _run("soporte climatizador", {"TPMS", "AA"}) is None


@pytest.mark.unit
def test_direct_route_support_word_matches_for_climatizador() -> None:
    assert _run("problema climatizador", {"TPMS", "AA", "CLIMATIZADOR"}) == "CLIMATIZADOR"
