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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import pipeline

app = FastAPI(title="RAG Demo")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# Serve the split-out CSS/JS (frontend/assets/*) at /assets/*.
app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")),
    name="assets",
)


class QueryRequest(BaseModel):
    query: str
    use_bm25: bool | None = None
    use_reranker: bool | None = None
    use_iterative: bool | None = None


@app.post("/query")
def query(req: QueryRequest):
    # Optional toggles let the Compare view drive the ablation live.
    if req.use_bm25 is not None:
        config.USE_BM25 = req.use_bm25
    if req.use_reranker is not None:
        config.USE_RERANKER = req.use_reranker
    if req.use_iterative is not None:
        config.USE_ITERATIVE = req.use_iterative
    # pipeline.answer_question must return {"answer": str, "chunks": [...]}
    return pipeline.answer_question(req.query)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
