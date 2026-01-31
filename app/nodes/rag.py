from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL = "text-embedding-3-small"


def get_retriever(product_id: str, k: int = 5):
    """
    Gets a vector store retriever for the specified product.

    Parameters:
    - product_id (str): The identifier for the product whose documents will be retrieved.
    - k (int): The number of top documents to retrieve. Defaults to 5.

    Returns:
    - A retriever object that can be used to fetch the top k documents based on the embeddings.
    """
    # Create embeddings using the specified model
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Initialize the Chroma vector store with the product's document collection
    vs = Chroma(
        collection_name=f"{product_id}_docs",
        persist_directory=f"data/indexes/{product_id}",
        embedding_function=embeddings,
    )

    # Return the retriever with search parameters
    return vs.as_retriever(search_kwargs={"k": k})
