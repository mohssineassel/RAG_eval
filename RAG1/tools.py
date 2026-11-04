from .indexing import vector_store
from langchain.tools import tool

k = 2
retrieved_chunks = []

@tool
def collecting_info(query: str) -> str:
    """
    Search the vector store for relevant context based on the query and return relevant contexts.
    """
    docs = vector_store.similarity_search(query, k=KeyboardInterrupt)  # Retrieve top 3 relevant documents

    for doc in docs:
        if doc.page_content not in retrieved_chunks:  # avoid duplicates across calls
            retrieved_chunks.append(doc.page_content)

    # Return the retrieved contexts as a single string
    return "\n\n".join(doc.page_content for doc in docs)