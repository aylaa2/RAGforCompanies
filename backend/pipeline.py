"""
Pipeline: the orchestrator — the backbone of the project.

Reads the toggles in config.py, decides which retrieval stages to run,
then hands the final chunks to generation. Keep it linear and readable.

The toggles are what the Compare view drives: the same question is run with
different (USE_BM25, USE_RERANKER) combinations to show the ablation.
"""

import config
import retrieval
import generate


def answer_question(query: str) -> dict:
    """
    Run the pipeline for one question, honoring the config toggles.
    Returns {"answer": str, "chunks": list[dict]}  (chunks kept for citations).
    """
    # 1. Dense semantic search always runs (the baseline).
    results = retrieval.semantic_search(query, config.TOP_K_RETRIEVE)

    # 2. Optionally add keyword search and fuse the two ranked lists (RRF).
    if config.USE_BM25:
        bm25 = retrieval.bm25_search(query, config.TOP_K_RETRIEVE)
        results = retrieval.fuse(results, bm25)

    # 3. Optionally rerank with the cross-encoder, else just truncate.
    if config.USE_RERANKER:
        results = retrieval.rerank(query, results, config.TOP_K_FINAL)
    else:
        results = results[: config.TOP_K_FINAL]

    # 4. Generate the grounded answer from the final chunks.
    answer = generate.generate_answer(query, results)
    return {"answer": answer, "chunks": results}
