from pathlib import Path
import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


def load_cfg():
    return yaml.safe_load(Path("config/routes.yaml").read_text(encoding="utf-8"))


def init_llm(model: str = "gpt-4o-mini", temperature: float = 0):
    return init_chat_model(model=model, temperature=temperature)
