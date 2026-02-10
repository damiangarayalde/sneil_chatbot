from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL = "text-embedding-3-small"


def _repo_root() -> Path:
    # rag.py is expected at: <root>/app/core/graph/nodes/rag.py
    return Path(__file__).resolve().parents[4]


def get_retriever(product_id: str, k: int = 5):
    """Return a Chroma retriever for a given product.

    Expects indexes at: <repo_root>/data/indexes/<product_id>
    Collection name: <product_id>_docs

    Notes:
    - If a caller mistakenly passes a collection name like "TPMS_docs",
      we will still work by mapping it back to product_id="TPMS".
    """
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Support accidental usage where product_id is actually the collection name
    if product_id.endswith("_docs"):
        collection_name = product_id
        base_product_id = product_id[: -len("_docs")]
    else:
        base_product_id = product_id
        collection_name = f"{product_id}_docs"

    persist_dir = _repo_root() / "data" / "indexes" / base_product_id

    # Avoid silently creating a new empty index in the wrong working directory.
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Chroma index folder not found: {persist_dir} (product_id={base_product_id})."
        )

    vs = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    return vs.as_retriever(search_kwargs={"k": k})
