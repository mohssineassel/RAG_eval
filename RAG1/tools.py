from .indexing import vector_store
from langchain.tools import tool

last_retrieved_context = [] 

@tool
def collecting_info(query: str) -> str:
    """
    Search the vector store for relevant context based on the query and return relevant contexts.
    """
    global last_retrieved_context
    docs = vector_store.similarity_search(query, k=3)  # Retrieve top 3 relevant documents

    # save the retrieved contexts for later use
    last_retrieved_context = [doc.page_content for doc in docs]

    # Return the retrieved contexts as a single string
    return "\n\n".join(last_retrieved_context)