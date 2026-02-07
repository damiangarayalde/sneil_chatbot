from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from app.utils import load_cfg


def load_prompt(path: str) -> str:
    """Load a prompt file from disk."""
    return Path(path).read_text(encoding="utf-8")


def get_shared_texts() -> dict:
    """Load all shared prompt texts used across routes."""
    BASE = load_prompt("app/prompts/shared/base_policy.md")
    WHATSAPP = load_prompt("app/prompts/shared/whatsapp_format.md")
    ESCALATION = load_prompt("app/prompts/shared/escalation_policy.md")
    SHIP_SPAIN = load_prompt("app/prompts/shared/shipping_spain.md")
    return {
        "BASE": BASE,
        "WHATSAPP": WHATSAPP,
        "ESCALATION": ESCALATION,
        "SHIP_SPAIN": SHIP_SPAIN,
    }


def make_system_text(route_id: str, route_prompt: str, max_chars: int, shared_texts: dict | None = None) -> str:
    """Compose the system message from route config and shared texts."""
    if shared_texts is None:
        shared_texts = get_shared_texts()
    # Some route prompt files include JSON examples with braces (e.g. { ... }).
    # Python's `str.format` used later will attempt to interpolate those braces,
    # causing KeyError. Escape braces in the route prompt so the text is treated
    # literally when the final ChatPromptTemplate formats messages.
    safe_route_prompt = route_prompt.replace("{", "{{").replace("}", "}}")

    return (
        shared_texts["BASE"]
        + "\n\n"
        # + shared_texts["WHATSAPP"]
        # + "\n\n"
        # + shared_texts["SHIP_SPAIN"]
        # + "\n\n"
        # + shared_texts["ESCALATION"]
        # + "\n\n"
        + f"## ROUTE: {route_id}\n"
        + safe_route_prompt
        + f"\n\nHard limit: {max_chars} characters including spaces."
    )


def make_chat_prompt(route_id: str, route_prompt: str, max_chars: int, human_template: str) -> ChatPromptTemplate:
    """Create a ChatPromptTemplate from system and human templates."""
    system = make_system_text(route_id, route_prompt, max_chars)
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human_template),
    ])


def get_route_config(route_id: str) -> tuple:
    """Load route configuration including prompt file and settings."""
    cfg = load_cfg()
    route_cfg = cfg.get(route_id)
    if route_cfg is None:
        raise KeyError(f"Route config not found: {route_id}")
    route_prompt = load_prompt(route_cfg["prompt_file"])
    max_chars = route_cfg["max_chars"]
    handoff_after = route_cfg.get("handoff_after_attempts")
    return route_prompt, max_chars, handoff_after, route_cfg


# def get_default_human_template(route_id: str) -> str:
#     """Default human template per route_id.

#     Centralizes input formatting so callers don't duplicate templates.
#     """
#     if route_id == "CLASSIFIER":
#         # ✅ This is the block you said you want to keep.
#         return (
#             "Historial reciente (puede estar vacío):\n{history}\n\n"
#             "Intentos de ruteo hasta ahora: {routing_attempts}\n"
#             "Resumen de triage actual (puede estar vacío): {triage_summary}\n"
#             "Sender (si existe): {from}\n\n"
#             "Último mensaje del usuario:\n{user_text}"
#         )

#     # Default for route modules (safe generic)
#     return "User: {user_text}"

# def make_chat_prompt_for_route(route_id: str, human_template: str | None = None) -> tuple[ChatPromptTemplate, dict]:
#     """Create a ChatPromptTemplate for a given route id.

#     Returns: (template, route_cfg)

#     If human_template is None, a default is selected based on route_id.
#     """

def make_chat_prompt_for_route(route_id: str, human_template: str) -> tuple[ChatPromptTemplate, dict]:
    """Create a ChatPromptTemplate for a given route id and return it
    along with the raw route config dict.

    This lets callers simply pass a `route_id` and receive a ready-to-use
    prompt template; internal prompt file loading and shared-text
    composition are handled here.
    """
    route_prompt, max_chars, handoff_after, route_cfg = get_route_config(
        route_id)

    # if human_template is None:
    #     human_template = get_default_human_template(route_id)

    tpl = make_chat_prompt(route_id, route_prompt, max_chars, human_template)
    return tpl, route_cfg
