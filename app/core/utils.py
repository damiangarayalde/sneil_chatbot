from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


# ---- Config loading (routes) ----
_CFG_CANDIDATES = [
    Path("app/core/config/config.yaml"),
    Path("app/core/config/config.yml"),
]


def load_cfg() -> Dict[str, Any]:
    """Load routing configuration from config/routes.(yaml|yml)."""
    for p in _CFG_CANDIDATES:
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raise FileNotFoundError(
        "Could not find config/routes.yaml or config/routes.yml"
    )


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

# ---- LLM init ----


def init_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    return init_chat_model(model=model, temperature=temperature)
