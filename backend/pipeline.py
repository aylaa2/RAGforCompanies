"""
Pipeline: the orchestrator.

Reads the toggles in config.py and decides which stages to run:
  retrieval:  semantic -> (+BM25 fuse) -> (+reranker)
  generation: single-pass, OR an iterative self-correction loop.

The toggles are what the Compare view and the evaluation drive, to show the
ablation (Semantic / +BM25 / +Reranker / +Iterative).

Returns {"answer": str, "chunks": list[dict]}.
"""

import config
import retrieval
import generate


def retrieve_chunks(query: str) -> list[dict]:
    """Run the retrieval stages for one query, honoring the config toggles."""
    results = retrieval.semantic_search(query, config.TOP_K_RETRIEVE)
    if config.USE_BM25:
        bm25 = retrieval.bm25_search(query, config.TOP_K_RETRIEVE)
        results = retrieval.fuse(results, bm25)
    if config.USE_RERANKER:
        results = retrieval.rerank(query, results, config.TOP_K_FINAL)
    else:
        results = results[: config.TOP_K_FINAL]
    return results


def _wants_search(answer: str):
    """If the model asked for more context ('CAUTĂ: X'), return X, else None."""
    cleaned = answer.strip().lstrip("*").strip()
    if cleaned.upper().startswith("CAUTĂ:"):
        return cleaned.split(":", 1)[1].strip()
    return None


def _answer_single(query: str) -> dict:
    chunks = retrieve_chunks(query)
    answer = generate.generate_answer(query, chunks)
    return {"answer": answer, "chunks": chunks}


def _answer_iterative(query: str, max_iterations: int) -> dict:
    """
    Corrective RAG: retrieve, let the model answer or ask for more context
    ('CAUTĂ: X'); if it asks, search X, accumulate chunks, and retry.
    """
    accumulated = {}
    search_query = query
    for i in range(max_iterations):
        for chunk in retrieve_chunks(search_query):
            accumulated.setdefault(chunk["id"], chunk)
        context = list(accumulated.values())

        answer = generate.generate_answer(query, context, allow_search=True)
        more = _wants_search(answer)
        if more is None:
            return {"answer": answer, "chunks": context}
        search_query = more  # loop again with the refined query

    # Fallback: hit the iteration limit — force a clean final answer (no CAUTĂ),
    # so we never return a raw "CAUTĂ: ..." string to the user.
    context = list(accumulated.values())
    answer = generate.generate_answer(query, context, allow_search=False)
    return {"answer": answer, "chunks": context}


def answer_question(query: str) -> dict:
    if config.USE_ITERATIVE:
        return _answer_iterative(query, config.ITERATIVE_MAX_ITERS)
    return _answer_single(query)
