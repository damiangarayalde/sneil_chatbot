from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# --------------------------------------------------------------------------------------
# Config loading (routes)
#
# Convention:
# - Top-level keys are route IDs (e.g., TPMS, AA, etc.)
# - 'CLASSIFIER' is a special section (not a route)
#
# You can override the config file location by setting one of:
# - ROUTES_CONFIG_PATH
# - ROUTES_CFG_PATH
# --------------------------------------------------------------------------------------

_CFG_CANDIDATES: List[Path] = [
    Path("app/core/config/config.yaml"),
    Path("app/core/config/config.yml"),
]

_CFG_ENV_VARS = ("ROUTES_CONFIG_PATH", "ROUTES_CFG_PATH")


def _candidate_paths() -> List[Path]:
    # 1) Explicit path overrides
    for env_var in _CFG_ENV_VARS:
        raw = os.getenv(env_var, "").strip()
        if raw:
            p = Path(raw).expanduser()
            return [p]

    # 2) Default repo paths
    return list(_CFG_CANDIDATES)


@lru_cache(maxsize=1)
def _load_cfg_cached() -> Dict[str, Any]:
    for p in _candidate_paths():
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raise FileNotFoundError(
        "Could not find routing config. Tried: "
        + ", ".join(str(p) for p in _candidate_paths())
        + ". You can also set ROUTES_CONFIG_PATH / ROUTES_CFG_PATH."
    )


def load_cfg() -> Dict[str, Any]:
    """Load routing configuration from config.yaml.

    This is cached (process-wide). Use `reload_cfg()` if you change the file on disk.
    """
    return _load_cfg_cached()


def reload_cfg() -> None:
    """Clear the in-process config cache (next `load_cfg()` will re-read from disk)."""
    _load_cfg_cached.cache_clear()


def get_routes(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return the list of enabled route IDs from config.

    Convention: top-level keys are route IDs; 'CLASSIFIER' is a special section.
    """
    cfg = cfg or load_cfg()
    return [k for k in cfg.keys() if k != "CLASSIFIER"]


def is_valid_route(route_id: Optional[str], cfg: Optional[Dict[str, Any]] = None) -> bool:
    if not route_id:
        return False
    cfg = cfg or load_cfg()
    return route_id in get_routes(cfg)


def get_route_section(route_id: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the config dict for a specific route_id (or empty dict)."""
    cfg = cfg or load_cfg()
    section = cfg.get(route_id, {}) or {}
    return section if isinstance(section, dict) else {}


def get_route_heuristics(route_id: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the optional heuristics config for a route.

    Supported config shapes (all optional):
    route_id:
      heuristics:
        mentions: [...]
        synonyms: [...]
        aliases: [...]
        clarifying_question: "..."
    """
    section = get_route_section(route_id, cfg)
    heur = section.get("heuristics") or section.get("heuristic") or {}
    return heur if isinstance(heur, dict) else {}


def get_route_mentions(route_id: str, cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return normalized alias strings used for cheap keyword routing."""
    heur = get_route_heuristics(route_id, cfg)
    raw_terms: List[str] = []

    for key in ("mentions", "synonyms", "aliases", "keywords"):
        val = heur.get(key)
        if isinstance(val, str) and val.strip():
            raw_terms.append(val)
        elif isinstance(val, list):
            raw_terms.extend(str(x) for x in val if str(x).strip())

    # Normalize + de-dup preserving order
    seen = set()
    out: List[str] = []
    for t in raw_terms:
        nt = t.strip().lower()
        if nt and nt not in seen:
            seen.add(nt)
            out.append(nt)
    return out


def get_route_clarifying_question(route_id: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Return the configured clarifying/disambiguation question for a route (if any)."""
    heur = get_route_heuristics(route_id, cfg)
    for key in ("clarifying_question", "disambiguation_question", "clarifier", "question"):
        val = heur.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


# --------------------------------------------------------------------------------------
# LLM init
# --------------------------------------------------------------------------------------

def init_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    return init_chat_model(model=model, temperature=temperature)
