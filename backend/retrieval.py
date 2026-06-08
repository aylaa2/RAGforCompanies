"""
Retrieval: all four retrieval-stage functions.

    semantic_search   -> dense vector search (always runs)
    bm25_search       -> keyword search (optional)
    fuse              -> combine ranked lists with RRF (optional)
    rerank            -> cross-encoder reorders candidates (optional)

All return the same dict shape: {"id", "text", "source", "score"}
so the stages compose cleanly.
"""

import re

import config

# ---- lazily-loaded, reused singletons (models + Chroma handles are heavy) ----
_embedder_model = None
_reranker_model = None
_collection = None
_bm25 = None
_bm25_docs = None


def _embedder():
    global _embedder_model
    if _embedder_model is None:
        from sentence_transformers import SentenceTransformer
        _embedder_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder_model


def _cross_encoder():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(config.RERANKER_MODEL)
    return _reranker_model


def _coll():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=config.STORAGE_DIR)
        _collection = client.get_collection(config.COLLECTION_NAME)
    return _collection


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def semantic_search(query: str, top_k: int) -> list[dict]:
    """Dense retrieval against Chroma (cosine over normalized e5 vectors)."""
    # e5 wants the query prefixed with "query: "
    q_emb = _embedder().encode("query: " + query, normalize_embeddings=True).tolist()
    res = _coll().query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i].get("source", "?"),
            "score": 1.0 - res["distances"][0][i],   # cosine distance -> similarity
        })
    return out


def _bm25_index():
    """Build the in-memory BM25 index once from all chunks stored in Chroma."""
    global _bm25, _bm25_docs
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        data = _coll().get(include=["documents", "metadatas"])
        _bm25_docs = [{
            "id": data["ids"][i],
            "text": data["documents"][i],
            "source": data["metadatas"][i].get("source", "?"),
        } for i in range(len(data["ids"]))]
        _bm25 = BM25Okapi([_tokenize(d["text"]) for d in _bm25_docs])
    return _bm25, _bm25_docs


def bm25_search(query: str, top_k: int) -> list[dict]:
    """Keyword retrieval with rank_bm25, in-memory and separate from Chroma."""
    bm25, docs = _bm25_index()
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{
        "id": docs[i]["id"],
        "text": docs[i]["text"],
        "source": docs[i]["source"],
        "score": float(scores[i]),
    } for i in order]


def fuse(semantic_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion.
    score(item) = sum over each list of 1 / (k + rank_in_that_list)
    """
    pool, fused_score = {}, {}
    for results in (semantic_results, bm25_results):
        for rank, item in enumerate(results):
            pool[item["id"]] = item
            fused_score[item["id"]] = fused_score.get(item["id"], 0.0) + 1.0 / (k + rank + 1)
    fused = []
    for id_, score in fused_score.items():
        item = dict(pool[id_])
        item["score"] = score
        fused.append(item)
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Cross-encoder reranking — scores (query, chunk) pairs together."""
    if not candidates:
        return []
    scores = _cross_encoder().predict([(query, c["text"]) for c in candidates])
    out = []
    for c, s in zip(candidates, scores):
        item = dict(c)
        item["score"] = float(s)
        out.append(item)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]
