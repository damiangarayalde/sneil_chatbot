from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL = "text-embedding-3-small"


def build_index(product_id: str):
    """
    Builds an index for the specified product by processing PDF documents.

    Parameters:
    - product_id (str): The identifier for the product whose PDFs will be indexed.

    This function loads PDF files from the specified product's directory, splits the text into manageable chunks,
    and stores the resulting index in a designated directory.
    """
    # Define the directory containing PDF files for the specified product
    pdf_dir = Path(f"knowledge/{product_id}/pdfs")

    # Define the directory where the index will be persisted
    persist_dir = Path(f"data/indexes/{product_id}")

    # Create the persist directory if it does not exist
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = []  # List to hold loaded documents

    # Load all PDF files from the product's PDF directory
    for pdf in pdf_dir.glob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf)).load())

    # Initialize the text splitter to divide documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150)

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=f"{product_id}_docs",
    ).persist()


# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # loop through a predefined list of products to build their indexes
    for product in ["TPMS", "AA", "GENKI", "CLIMATIZADOR", "CARJACK", "MAYORISTA", "CALDERA"]:
        build_index(product)
