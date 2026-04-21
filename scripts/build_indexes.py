from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "text-embedding-3-small"

# run with: python scripts/build_indexes.py --products TPMS AA --rebuild


def _find_repo_root(start: Path) -> Path:
    """Find the repository root even if this script is executed from another folder."""
    for p in [start, *start.parents]:
        # Strong-ish layout markers for this project
        if (p / "knowledge").exists() and (p / "data").exists():
            return p
        if (p / "app").exists() and (p / "scripts").exists():
            return p
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return start


# Load environment variables (expects OPENAI_API_KEY, etc.)
REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(override=False)


def build_index(product_id: str):
    """Build a Chroma index for a product.

    Expected layout:
      <root>/knowledge/<product_id>/*.docx
      <root>/data/indexes/<product_id>/
    """
    print(f"\n{'='*60}")
    print(f"Building index for: {product_id}")
    print(f"REPO_ROOT: {REPO_ROOT}")
    print(f"{'='*60}\n")

    docx_dir = REPO_ROOT / "knowledge" / product_id
    persist_dir = REPO_ROOT / "data" / "indexes" / product_id

    print(f"📂 Loading DOCX files from: {docx_dir}")
    if not docx_dir.exists():
        print(f"❌ Missing knowledge folder: {docx_dir}")
        return

    docx_files = list(docx_dir.glob("*.docx"))
    print(f"   Found {len(docx_files)} file(s)\n")

    if not docx_files:
        print("❌ No .docx files found. Index would be empty; aborting.")
        return

    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    for i, docx in enumerate(docx_files, 1):
        print(f"   [{i}/{len(docx_files)}] Loading: {docx.name}...",
              end=" ", flush=True)
        loaded = Docx2txtLoader(str(docx)).load()
        for doc in loaded:
            doc.metadata["doc_name"] = docx.stem
        docs.extend(loaded)
        print("✓")

    print(f"\n✅ Loaded {len(docs)} documents\n")
    if not docs:
        print("❌ Loaded 0 documents. Index would be empty; aborting.")
        return

    print("✂️  Splitting documents into chunks (size: 900, overlap: 150)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks\n")

    if not chunks:
        print("❌ Created 0 chunks. Index would be empty; aborting.")
        return

    print("🔄 Generating embeddings and building vector store...")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   Persist dir: {persist_dir}\n")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=f"{product_id}_docs",
    )

    # Persist explicitly for robustness across versions
    try:
        vs.persist()
    except Exception:
        pass

    # Best-effort validation
    try:
        count = vs._collection.count()  # type: ignore[attr-defined]
        print(f"📦 Persisted vectors count: {count}")
    except Exception:
        pass

    print(f"✅ Index built successfully for {product_id}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    for product in ["TPMS", "AA", "CLIMATIZADORES", "CALDERAS"]:
        build_index(product)
