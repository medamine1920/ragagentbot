import os

class Config:
    OLLAMA_MODEL = "llama3"  # or "mistral", "phi3", etc.
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200