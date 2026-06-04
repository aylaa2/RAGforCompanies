"""
Retrieval: all four retrieval-stage functions.

    semantic_search   -> dense vector search (always runs)
    bm25_search       -> keyword search (optional)
    fuse              -> combine ranked lists with RRF (optional)
    rerank            -> cross-encoder reorders candidates (optional)

All return the same dict shape: {"id", "text", "source", "score"}
so the stages compose cleanly.
"""

import config


def semantic_search(query: str, top_k: int) -> list[dict]:
    """
    Dense retrieval against Chroma.

    TODO:
      - embed the query (same model as ingest)
      - open persistent Chroma client + collection
      - query top_k nearest chunks
      - normalise into the dict shape above
    """
    raise NotImplementedError


def bm25_search(query: str, top_k: int) -> list[dict]:
    """
    Keyword retrieval with rank_bm25.

    TODO:
      - hold the tokenised chunk corpus in memory to build BM25Okapi
      - score the query, take top_k
    NOTE: in-memory, separate from Chroma. Chroma does only the semantic
          half; fusion happens in fuse().
    """
    raise NotImplementedError


def fuse(semantic_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion.
    score(item) = sum over each list of 1 / (k + rank_in_that_list)

    TODO (~10 lines — write it yourself, easy to explain in Q&A):
      - build rank maps per list (by id)
      - sum 1/(k+rank) across lists for every unique id
      - return sorted by score desc
    """
    raise NotImplementedError


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Cross-encoder reranking — scores (query, chunk) pairs together.

    TODO:
      - load cross-encoder (config.RERANKER_MODEL)
      - score each (query, candidate.text) pair
      - sort by score, return top_k
    """
    raise NotImplementedError
