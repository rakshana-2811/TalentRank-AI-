from dotenv import load_dotenv
import os
from openai import OpenAI
import numpy as np
from typing import List


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "").lower() == "true"

# Initialize OpenAI client if available
_client = None
if OPENAI_API_KEY and not USE_LOCAL_EMBEDDINGS:
    _client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize local model if requested
_local_model = None
if USE_LOCAL_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        raise ImportError(
            "USE_LOCAL_EMBEDDINGS=true but sentence_transformers not installed. "
            "Run: pip install sentence-transformers"
        )


def get_embedding(text: str) -> np.ndarray:
    """Return embedding vector for a single `text`.

    Uses OpenAI (if key set) or local sentence-transformers (if USE_LOCAL_EMBEDDINGS=true).
    """
    if USE_LOCAL_EMBEDDINGS:
        if _local_model is None:
            raise RuntimeError("Local model not initialized. Check sentence_transformers installation.")
        vec = _local_model.encode(text, convert_to_numpy=True)
        return np.array(vec, dtype=np.float32)
    else:
        if not _client or not OPENAI_API_KEY:
            raise RuntimeError(
                "OpenAI key not set. Either set OPENAI_API_KEY or use USE_LOCAL_EMBEDDINGS=true"
            )
        resp = _client.embeddings.create(model="text-embedding-3-small", input=text)
        vec = resp["data"][0]["embedding"]
        return np.array(vec, dtype=np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts and return a (n, dim) numpy array."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    
    if USE_LOCAL_EMBEDDINGS:
        if _local_model is None:
            raise RuntimeError("Local model not initialized. Check sentence_transformers installation.")
        vecs = _local_model.encode(texts, convert_to_numpy=True)
        return np.array(vecs, dtype=np.float32)
    else:
        if not _client or not OPENAI_API_KEY:
            raise RuntimeError(
                "OpenAI key not set. Either set OPENAI_API_KEY or use USE_LOCAL_EMBEDDINGS=true"
            )
        resp = _client.embeddings.create(model="text-embedding-3-small", input=texts)
        vectors = [item["embedding"] for item in resp["data"]]
        return np.array(vectors, dtype=np.float32)


def similarity_score(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between embeddings of two texts.

    This function requests embeddings for `text_a` and `text_b` from OpenAI
    and returns their cosine similarity as a float in [-1.0, 1.0].

    Raises:
        RuntimeError: if `OPENAI_API_KEY` is not set.
        Exception: for other errors from the OpenAI SDK.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Load .env or set env var.")

    # Import locally to avoid circular imports at module-import time.
    from .similarity import cosine_similarity

    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)

    return float(cosine_similarity(emb_a, emb_b))
