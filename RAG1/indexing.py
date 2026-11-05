from dotenv import load_dotenv
from .vectorstore import build_vector_store
import os


load_dotenv()  # Load environment variables from .env file


vector_store = build_vector_store(os.getenv("EMBEDDING_PROVIDER", "openai"))
