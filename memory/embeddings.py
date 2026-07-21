"""Cached Hugging Face embeddings for semantic memory."""
from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    """Load the 384-dimensional embedding model once per process."""
    try: return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except OSError: return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed text batches for LangGraph's semantic store index."""
    return model().encode(texts, normalize_embeddings=True).tolist()
