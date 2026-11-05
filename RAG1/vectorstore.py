from .file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from pathlib import Path

PDF_PATH = Path(__file__).resolve().parent.parent / "pdfs" / "Penguins_ACL.pdf"  


def get_embedding(provider: str):
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")  
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")  
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-large")


def build_vector_store(provider: str) -> InMemoryVectorStore:
    """Embed the PDF paragraphs with the given embedding provider."""
    vector_store = InMemoryVectorStore(get_embedding(provider))  
    chunks = extract_paragraphs(PDF_PATH)  
    vector_store.add_texts(texts=chunks)  
    print(f"Added {len(chunks)} chunks to the vector store ({provider} embeddings).")
    return vector_store
