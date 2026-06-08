# RAG Demo

Document-Q&A RAG built as an ablation: semantic search → hybrid (+ BM25)
→ reranker. Each stage is a toggle, so you can show side by side how
retrieval quality improves as components are added.

FastAPI serves both the API and a small built-in UI. One server, no Node.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Add an LLM client to `requirements.txt` and set `LLM_MODEL` in `backend/config.py`.

## Usage

1. Put source documents in `data/documents/`
2. Build the vector store (run once, from repo root):
   ```bash
   python backend/ingest.py
   ```
3. Start the server (from repo root):
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
4. Open http://localhost:8000

## What's already done vs. your job

DONE (don't touch):
  - `backend/main.py`     — FastAPI app, /query endpoint, serves the UI
  - `backend/static/`     — the chat UI (answer + citations + toggles)

YOUR JOB (the RAG pipeline — these are stubbed with TODOs):
  - `backend/ingest.py`     load → chunk → embed → store in Chroma
  - `backend/retrieval.py`  semantic, bm25, fuse, rerank
  - `backend/generate.py`   build prompt + call the LLM
  - `backend/pipeline.py`   orchestrator that ties the stages together
  - `backend/config.py`     set your model names

## Suggested build order

Get an end-to-end path working before adding stages. Commit after each step.

1. `ingest.py` + `semantic_search()` + `generate_answer()` + `pipeline.py`
   wired to semantic only. Run the server, ask one question, get one answer.
2. `bm25_search()` + `fuse()` — flip BM25 on in the UI.
3. `rerank()` — flip reranker on in the UI.

## Future work ("where this goes" slide)

- **Agentic self-correction (Corrective RAG):** grade retrieved chunks, and
  if weak, rewrite the query and retry once. One more rung after the reranker.
- **GraphRAG:** entity/relationship graph at ingestion for global, thematic
  questions. A parallel retrieval system, not a stage — heavier lift.
