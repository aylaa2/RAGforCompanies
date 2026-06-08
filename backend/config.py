"""
Central configuration for the RAG pipeline.

The TOGGLES below are the heart of the ablation demo:
flip them, re-run the same questions, show how retrieval quality
changes as each stage is added.
"""

import os

USE_BM25 = True          # add keyword search + fuse with semantic (RRF)
USE_RERANKER = True      # add cross-encoder reranking on the fused list
USE_ITERATIVE = True     # iterative self-correction loop (model can ask "CAUTĂ: X")
ITERATIVE_MAX_ITERS = 3  # max retrieval rounds in the iterative loop

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"    # sentence-transformers, multilingual
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # light multilingual cross-encoder (fast)
LLM_MODEL = "qwen2.5:3b"                             # local, via Ollama (ollama pull qwen2.5:3b)

TOP_K_RETRIEVE = 6      # how many chunks each retriever pulls (lower = faster rerank)
TOP_K_FINAL = 4         # how many chunks to feed the LLM after reranking

CHUNK_SIZE = 500        # characters (or tokens — your call)
CHUNK_OVERLAP = 50

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(_PROJECT_ROOT, "data", "documents")
STORAGE_DIR = os.path.join(_PROJECT_ROOT, "storage")
COLLECTION_NAME = "rag_demo"
