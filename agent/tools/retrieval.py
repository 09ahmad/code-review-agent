"""
Retrieval tool: a small FAISS-backed vector store over past review comments
/ known bug patterns, so the agent can say "this pattern caused issue X in
the past" instead of reasoning from scratch every time.

This is intentionally simple (in-memory FAISS) so it runs anywhere without
a hosted vector DB — swap in Pinecone/Weaviate/pgvector as a "next step" if
you want to talk about scaling it in an interview.

Robustness notes:
- sentence_transformers/faiss are imported LAZILY (inside functions, not at
  module load time) so that if either is missing or fails to load in a given
  environment, only this one tool degrades -- the rest of the app still
  starts and runs normally.
- The embedding model + FAISS index are built ONCE and cached at module
  level, instead of being rebuilt on every call -- important for a live
  demo where re-loading a model on every click would look slow and broken.
"""

from __future__ import annotations

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# A handful of seed "past bug patterns" so this runs out of the box.
# Replace with real historical review comments from your own repos.
DEFAULT_CORPUS = [
    "String interpolation used to build a SQL query — SQL injection risk.",
    "Division without checking for zero denominator.",
    "Database commit called inside a loop instead of batched.",
    "Mutable default argument used in a function signature.",
    "Broad except clause swallowing all exceptions silently.",
    "Hardcoded credentials or API keys committed to source.",
    "Missing null/None check before attribute access.",
    "Off-by-one error in loop bound (range vs len usage).",
]

# Module-level cache: built once, reused across every call in this process.
_retriever_cache = None
_retriever_unavailable_reason = None


class ReviewRetriever:
    def __init__(self, corpus=None):
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        self._np = np
        self.corpus = corpus or DEFAULT_CORPUS
        self.embedder = SentenceTransformer(_EMBED_MODEL_NAME)
        embeddings = self.embedder.encode(self.corpus, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype="float32"))

    def retrieve(self, query, k=3):
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(
            self._np.array(query_vec, dtype="float32"), k
        )
        return [self.corpus[i] for i in indices[0] if i < len(self.corpus)]


def _get_retriever():
    """Lazily builds and caches the retriever. Returns None if unavailable."""
    global _retriever_cache, _retriever_unavailable_reason

    if _retriever_cache is not None:
        return _retriever_cache
    if _retriever_unavailable_reason is not None:
        return None

    try:
        _retriever_cache = ReviewRetriever()
        return _retriever_cache
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see module note
        _retriever_unavailable_reason = "{}: {}".format(type(exc).__name__, exc)
        return None


def retrieve_similar_patterns(diff, k=3):
    """
    Convenience function used by the graph nodes. Returns an empty list
    (never raises) if the retrieval backend isn't available in this
    environment -- the rest of the review still proceeds normally.
    """
    retriever = _get_retriever()
    if retriever is None:
        return []
    return retriever.retrieve(diff, k=k)
