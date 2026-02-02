from pathlib import Path
import yaml
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()


def load_cfg():
    return yaml.safe_load(Path("config/routes.yaml").read_text(encoding="utf-8"))


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def get_shared_texts():
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
    if shared_texts is None:
        shared_texts = get_shared_texts()
    return (
        shared_texts["BASE"]
        # + "\n\n"
        # + shared_texts["WHATSAPP"]
        # + "\n\n"
        # + shared_texts["SHIP_SPAIN"]
        # + "\n\n"
        # + shared_texts["ESCALATION"]
        # + "\n\n"
        # + f"## ROUTE: {route_id}\n"
        # + route_prompt
        # + f"\n\nHard limit: {max_chars} characters including spaces."
    )


def make_chat_prompt(route_id: str, route_prompt: str, max_chars: int, human_template: str) -> ChatPromptTemplate:
    system = make_system_text(route_id, route_prompt, max_chars)
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human_template),
    ])


def init_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    return init_chat_model(model=model, temperature=temperature)


def get_route_config(route_id: str):
    cfg = load_cfg()
    route_cfg = cfg.get(route_id)
    if route_cfg is None:
        raise KeyError(f"Route config not found: {route_id}")
    route_prompt = load_prompt(route_cfg["prompt_file"])
    max_chars = route_cfg["max_chars"]
    handoff_after = route_cfg.get("handoff_after_attempts")
    return route_prompt, max_chars, handoff_after, route_cfg


def make_chat_prompt_for_route(route_id: str, human_template: str) -> tuple[ChatPromptTemplate, dict]:
    """Create a ChatPromptTemplate for a given route id and return it
    along with the raw route config dict.

    This lets callers simply pass a `route_id` and receive a ready-to-use
    prompt template; internal prompt file loading and shared-text
    composition are handled here.
    """
    route_prompt, max_chars, handoff_after, route_cfg = get_route_config(
        route_id)
    tpl = make_chat_prompt(route_id, route_prompt, max_chars, human_template)
    return tpl, route_cfg
