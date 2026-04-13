"""
Unit tests for the config schema (Gap 8 — Config validation).

All tests are pure Python — no LLM, no network.
The real config.yaml is read from disk; no fixtures needed.

Run:
    pytest -m unit
    pytest tests/unit/test_config_schema.py -v
"""
from __future__ import annotations

import pytest

from app.core.config.schema import AppConfig, ClassifierConfig, RouteConfig, validate_cfg
from app.core.utils import load_cfg


# ---------------------------------------------------------------------------
# Live config round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_real_config_passes_validation() -> None:
    """The live config.yaml must pass schema validation without errors."""
    raw = load_cfg()
    validate_cfg(raw)  # Raises on any violation


@pytest.mark.unit
def test_classifier_section_present() -> None:
    raw = load_cfg()
    assert "CLASSIFIER" in raw


@pytest.mark.unit
def test_all_routes_have_prompt_file() -> None:
    raw = load_cfg()
    for key, value in raw.items():
        if key == "CLASSIFIER" or not isinstance(value, dict):
            continue
        assert "prompt_file" in value, f"Route '{key}' is missing required key 'prompt_file'"


@pytest.mark.unit
def test_classifier_fields_are_within_bounds() -> None:
    raw = load_cfg()
    cfg = ClassifierConfig(**raw["CLASSIFIER"])
    assert cfg.max_chars > 0
    assert 0.0 <= cfg.route_lock_threshold <= 1.0
    assert cfg.max_attempts_before_handoff >= 1


@pytest.mark.unit
def test_all_route_max_attempts_at_least_one() -> None:
    raw = load_cfg()
    for key, value in raw.items():
        if key == "CLASSIFIER" or not isinstance(value, dict):
            continue
        cfg = RouteConfig(**value)
        assert cfg.max_attempts_before_handoff >= 1, (
            f"Route '{key}': max_attempts_before_handoff must be >= 1"
        )


# ---------------------------------------------------------------------------
# ClassifierConfig constraint violations
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("field, bad_value, error_fragment", [
    ("max_chars", 0, "greater than 0"),
    ("max_chars", -10, "greater than 0"),
    ("route_lock_threshold", 1.5, "less than or equal to 1"),
    ("route_lock_threshold", -0.1, "greater than or equal to 0"),
    ("max_attempts_before_handoff", 0, "greater than or equal to 1"),
])
def test_classifier_rejects_invalid_values(
    field: str, bad_value: object, error_fragment: str
) -> None:
    raw = load_cfg()
    data = dict(raw["CLASSIFIER"])
    data[field] = bad_value
    with pytest.raises(Exception, match=error_fragment):
        ClassifierConfig(**data)


# ---------------------------------------------------------------------------
# RouteConfig constraint violations
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("field, bad_value, error_fragment", [
    ("max_chars", 0, "greater than 0"),
    ("max_chars", -5, "greater than 0"),
    ("max_attempts_before_handoff", 0, "greater than or equal to 1"),
])
def test_route_rejects_invalid_values(
    field: str, bad_value: object, error_fragment: str
) -> None:
    raw = load_cfg()
    route_key = next(k for k in raw if k != "CLASSIFIER" and isinstance(raw[k], dict))
    data = dict(raw[route_key])
    data[field] = bad_value
    with pytest.raises(Exception, match=error_fragment):
        RouteConfig(**data)


# ---------------------------------------------------------------------------
# AppConfig / validate_cfg
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_cfg_raises_on_bad_classifier() -> None:
    bad_cfg = {
        "CLASSIFIER": {
            "max_chars": -1,
            "prompt_file": "some/path.md",
            "max_attempts_before_handoff": 3,
            "route_lock_threshold": 0.75,
        }
    }
    with pytest.raises(Exception):
        validate_cfg(bad_cfg)


@pytest.mark.unit
def test_validate_cfg_raises_when_classifier_missing() -> None:
    with pytest.raises(Exception):
        validate_cfg({})


@pytest.mark.unit
def test_validate_cfg_raises_on_bad_route() -> None:
    raw = load_cfg()
    bad_cfg = dict(raw)
    # Inject a route with an invalid max_chars
    bad_cfg["FAKE_ROUTE"] = {
        "max_chars": 0,
        "prompt_file": "some/path.md",
        "max_attempts_before_handoff": 1,
    }
    with pytest.raises(Exception, match="greater than 0"):
        validate_cfg(bad_cfg)


@pytest.mark.unit
def test_extra_route_keys_are_tolerated() -> None:
    """RouteConfig must not reject unknown keys — forward compatibility."""
    raw = load_cfg()
    route_key = next(k for k in raw if k != "CLASSIFIER" and isinstance(raw[k], dict))
    data = dict(raw[route_key])
    data["some_future_key"] = "value"
    RouteConfig(**data)  # Must not raise
