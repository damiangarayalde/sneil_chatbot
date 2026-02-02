from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv


EMBED_MODEL = "text-embedding-3-small"
load_dotenv()


def build_index(product_id: str):
    """
    Builds an index for the specified product by processing DOCX documents.

    Parameters:
    - product_id (str): The identifier for the product whose DOCX files will be indexed.

    This function loads DOCX files from the specified product's directory, splits the text into manageable chunks,
    and stores the resulting index in a designated directory.
    """
    print(f"\n{'='*60}")
    print(f"Building index for: {product_id}")
    print(f"{'='*60}\n")

    # Define the directory containing DOCX files for the specified product
    docx_dir = Path(f"knowledge/{product_id}")

    # Define the directory where the index will be persisted
    persist_dir = Path(f"data/indexes/{product_id}")

    # Create the persist directory if it does not exist
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = []  # List to hold loaded documents

    # Load all DOCX files from the product's directory
    print(f"📂 Loading DOCX files from: {docx_dir}")
    docx_files = list(docx_dir.glob("*.docx"))
    print(f"   Found {len(docx_files)} file(s)\n")

    for i, docx in enumerate(docx_files, 1):
        print(f"   [{i}/{len(docx_files)}] Loading: {docx.name}...",
              end=" ", flush=True)
        docs.extend(Docx2txtLoader(str(docx)).load())
        print("✓")

    print(f"\n✅ Loaded {len(docs)} documents\n")

    # Initialize the text splitter to divide documents into chunks
    print(f"✂️  Splitting documents into chunks (size: 900, overlap: 150)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150)

    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks\n")

    # Create embeddings and build vector store
    print(f"🔄 Generating embeddings and building vector store...")
    print(f"   Model: {EMBED_MODEL}")
    print(f"   Persist dir: {persist_dir}\n")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=f"{product_id}_docs",
    )

    print(f"✅ Index built successfully for {product_id}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # loop through a predefined list of products to build their indexes
    for product in ["TPMS", "AA"]:
        # , "GENKI", "CLIMATIZADOR", "CARJACK", "MAYORISTA", "CALDERA"]:
        build_index(product)
