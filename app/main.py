from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from app.graph import build_graph

load_dotenv()

graph = build_graph()

state = {
    "messages": [HumanMessage(content="Hola, tengo un c260 y no me aparecen 2 sensores")],
    "route": None,
    "attempts": {},
    "retrieved": None,
    "answer": None,
}

out = graph.invoke(state)
print(out["answer"])
