"""
Ingestion: run this ONCE to build the vector store.

    load documents -> chunk -> embed -> write to Chroma (persisted to disk)

NOT part of the live query path. Run from the repo root:
    python backend/ingest.py
"""

import config


def load_documents(documents_dir: str) -> list[dict]:
    """
    Load every document from documents_dir.
    Returns [{"id": ..., "text": ..., "source": ...}, ...]

    TODO:
      - decide supported file types (.txt, .md, .pdf?)
      - read each file's text
      - keep the filename as 'source' for citations later
    """
    raise NotImplementedError


def chunk_document(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split one document into overlapping chunks.

    TODO:
      - simplest: slice by character count with overlap
      - better: split on paragraphs/sentences, then pack to size
    """
    raise NotImplementedError


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Texts -> embedding vectors.

    TODO:
      - load sentence-transformers model (config.EMBEDDING_MODEL) ONCE
      - return model.encode(texts)
    """
    raise NotImplementedError


def main():
    """
    TODO — orchestrate:
      1. docs = load_documents(config.DOCUMENTS_DIR)
      2. chunk each doc (carry source + chunk index in metadata)
      3. embed all chunks
      4. open persistent Chroma client at config.STORAGE_DIR
      5. create/get collection config.COLLECTION_NAME
      6. add ids, embeddings, documents (text), metadatas
      7. print a count so you know it worked
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
