from dotenv import load_dotenv
from .file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
import os
from pathlib import Path


load_dotenv()  # Load environment variables from .env file

embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")

if embedding_provider == "gemini":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")  # free tier
elif embedding_provider == "ollama":
    from langchain_ollama import OllamaEmbeddings
    embedding = OllamaEmbeddings(model="nomic-embed-text")  # local, fast enough for embeddings
else:
    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embedding)  # Create an in-memory vector store

PDF_PATH = Path(__file__).resolve().parent.parent / "pdfs" / "Penguins_ACL.pdf"  # works from any folder

chunks = extract_paragraphs(PDF_PATH)  # Extract paragraphs from the PDF
vector_store.add_texts(texts=chunks)  # Add the extracted paragraphs to the vector store
print(f"Added {len(chunks)} chunks to the vector store.")  # Print the number of chunks added