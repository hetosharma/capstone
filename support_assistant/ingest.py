"""Create the local Chroma collection from the eight checked-in policy documents."""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "zepto_support_policies"


def ingest() -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    documents, ids, metadatas = [], [], []
    for path in sorted(DOCS.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        documents.append(text)
        ids.append(path.stem)
        metadatas.append({"source": path.stem, "filename": path.name})
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(ids)} documents in {COLLECTION}: {ids}")


if __name__ == "__main__":
    ingest()


