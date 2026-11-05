from dotenv import load_dotenv
from .file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
import os


load_dotenv()  # Load environment variables from .env file

if os.getenv("USE_LOCAL") == "true":
    from langchain_ollama import OllamaEmbeddings
    embedding = OllamaEmbeddings(model="nomic-embed-text")
else:
    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embedding)  # Create an in-memory vector store

chunks = extract_paragraphs("pdfs/Penguins_ACL.pdf")  # Extract paragraphs from the PDF
vector_store.add_texts(texts=chunks)  # Add the extracted paragraphs to the vector store
print(f"Added {len(chunks)} chunks to the vector store.")  # Print the number of chunks added