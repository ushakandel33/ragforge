"""
Thin wrapper around a persistent ChromaDB collection. Chosen because it
runs embedded (no separate server process needed), which keeps the Docker
image small and the local dev loop fast.
"""
import uuid
from functools import lru_cache
import chromadb
from app.config import settings
from app.rag.embeddings import embed_texts, embed_query

COLLECTION_NAME = "documents"


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=settings.vector_db_path)


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def add_chunks(chunks: list[dict]) -> int:
    """chunks: list of {"text": str, "metadata": dict}"""
    if not chunks:
        return 0
    collection = get_collection()
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embed_texts(texts)
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def query(text: str, top_k: int = None) -> list[dict]:
    top_k = top_k or settings.top_k_retrieval
    collection = get_collection()
    if collection.count() == 0:
        return []
    query_embedding = embed_query(text)
    results = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, collection.count()))

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        # Chroma returns distance (lower = closer); convert to a similarity-like score.
        hits.append({"text": doc, "metadata": meta, "score": 1 - dist})
    return hits
