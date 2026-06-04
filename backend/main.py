"""
FastAPI backend. Serves the static UI AND the RAG pipeline over HTTP.
One server, no Node, no CORS.

Run from the repo root:
    uvicorn backend.main:app --reload --port 8000

Then open http://localhost:8000

This file is DONE — you shouldn't need to edit it. It calls
pipeline.answer_question(), which is the part you implement.
"""

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
import pipeline

app = FastAPI(title="RAG Demo")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


class QueryRequest(BaseModel):
    query: str
    use_bm25: bool | None = None
    use_reranker: bool | None = None


@app.post("/query")
def query(req: QueryRequest):
    # UI toggles drive the ablation live.
    if req.use_bm25 is not None:
        config.USE_BM25 = req.use_bm25
    if req.use_reranker is not None:
        config.USE_RERANKER = req.use_reranker

    # pipeline.answer_question must return {"answer": str, "chunks": [...]}
    return pipeline.answer_question(req.query)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
