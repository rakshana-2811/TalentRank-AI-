import numpy as np
from typing import List, Tuple


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def rank_resumes(job_embedding: np.ndarray, resume_embeddings: np.ndarray, filenames: List[str]) -> List[Tuple[str, float]]:
    """Return list of (filename, score) sorted by descending similarity.

    `resume_embeddings` is expected to be shape (n, dim).
    """
    scores = []
    for i, fname in enumerate(filenames):
        emb = resume_embeddings[i]
        score = cosine_similarity(job_embedding, emb)
        scores.append((fname, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)
