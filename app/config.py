import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Local sentence-transformers model by default (no API key needed). Set a
# "models/..." name (e.g. models/text-embedding-004, models/gemini-embedding-001)
# to use the Gemini embedding API instead.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))  # all-MiniLM-L6-v2 output size (768 for text-embedding-004)

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 4))
# Cosine similarity cutoff below which we refuse to answer rather than let the LLM guess.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))

# --- Storage (local only, per task spec) ---
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
