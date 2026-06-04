"""
Pipeline: the orchestrator — the backbone of the project.

Reads the toggles in config.py, decides which retrieval stages to run,
then hands the final chunks to generation. Keep it linear and readable.
"""

import config
import retrieval
import generate


def answer_question(query: str) -> dict:
    """
    Run the full pipeline for one question.
    Returns {"answer": str, "chunks": list[dict]}  (chunks kept for citations)

    TODO — the linear flow:
      1. results = retrieval.semantic_search(query, config.TOP_K_RETRIEVE)   # always

      2. if config.USE_BM25:
             bm25 = retrieval.bm25_search(query, config.TOP_K_RETRIEVE)
             results = retrieval.fuse(results, bm25)

      3. if config.USE_RERANKER:
             results = retrieval.rerank(query, results, config.TOP_K_FINAL)
         else:
             results = results[: config.TOP_K_FINAL]

      4. answer = generate.generate_answer(query, results)
      5. return {"answer": answer, "chunks": results}
    """
    raise NotImplementedError
