"""
Ingestion: run this ONCE to build the vector store.

    load documents -> chunk -> embed -> write to Chroma (persisted to disk)

NOT part of the live query path. Run from the repo root:
    python backend/ingest.py
"""

import os

import config

# The embedding model is heavy; load it once and reuse.
_model = None


def _embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def load_documents(documents_dir: str) -> list[dict]:
    """
    Load every .md / .txt document from documents_dir.
    Returns [{"id": ..., "text": ..., "source": ...}, ...]
    The filename is kept as 'source' so we can cite it later.
    """
    docs = []
    for name in sorted(os.listdir(documents_dir)):
        if not name.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(documents_dir, name)
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append({"id": name, "text": text, "source": name})
    return docs


def chunk_document(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split one document into overlapping character chunks.
    Slide a window of chunk_size, stepping by (chunk_size - overlap) so
    consecutive chunks share `overlap` characters of context.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Texts -> normalized embedding vectors (cosine-ready).
    NOTE: callers add the e5 "passage: " / "query: " prefix themselves
    (see config.py) — this function embeds whatever it is given.
    """
    return _embedder().encode(texts, normalize_embeddings=True).tolist()


def main():
    docs = load_documents(config.DOCUMENTS_DIR)
    if not docs:
        print(f"Niciun document în {config.DOCUMENTS_DIR}. Adaugă fișiere .md/.txt și reia.")
        return

    ids, texts, metas = [], [], []
    for d in docs:
        for i, chunk in enumerate(chunk_document(d["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            ids.append(f"{d['source']}#{i}")
            texts.append(chunk)
            metas.append({"source": d["source"], "chunk": i})

    print(f"{len(docs)} documente -> {len(texts)} fragmente. Generez embeddings (prima rulare descarcă modelul)...")
    # e5 wants each indexed chunk prefixed with "passage: "
    embeddings = embed_texts(["passage: " + t for t in texts])

    import chromadb
    client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    # Rebuild cleanly so re-running doesn't duplicate chunks.
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)

    print(f"Gata. {collection.count()} fragmente în colecția '{config.COLLECTION_NAME}' la {config.STORAGE_DIR}/.")


if __name__ == "__main__":
    main()
