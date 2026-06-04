"""
Central configuration for the RAG pipeline.

The TOGGLES below are the heart of the ablation demo:
flip them, re-run the same questions, show how retrieval quality
changes as each stage is added.
"""

# --- Ablation toggles -------------------------------------------------
# Semantic (dense) search is always on — it's the baseline.
USE_BM25 = False        # add keyword search + fuse with semantic (RRF)
USE_RERANKER = False    # add cross-encoder reranking on the fused list

# --- Model names ------------------------------------------------------
# TODO: pick your models. Suggested starting points below.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"          # sentence-transformers
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "TODO"                                   # whatever you have access to

# --- Retrieval params -------------------------------------------------
TOP_K_RETRIEVE = 10     # how many chunks each retriever pulls
TOP_K_FINAL = 4         # how many chunks to feed the LLM after reranking

# --- Chunking params --------------------------------------------------
CHUNK_SIZE = 500        # characters (or tokens — your call)
CHUNK_OVERLAP = 50

# --- Paths (relative to repo root; run backend from there) ------------
DOCUMENTS_DIR = "data/documents"
STORAGE_DIR = "storage"
COLLECTION_NAME = "rag_demo"
