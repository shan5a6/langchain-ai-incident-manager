from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore  # Changed from Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings (globally)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_retriever(as_retriever: bool = True):
    """
    Returns a LangChain Qdrant VectorStore or Retriever
    """

    client = QdrantClient(
        url="http://qdrant:6333",
        timeout=10
    )

    # Use QdrantVectorStore with the explicit client
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="incidents",
        embedding=embeddings
    )

    if as_retriever:
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    return vectorstore
