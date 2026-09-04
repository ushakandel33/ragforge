"""
Embedding generation using a local sentence-transformers model. Running
embeddings locally (rather than via an API) means ingestion works fully
offline and costs nothing, keeping the RAG pipeline usable with only a
free LLM API key for the generation step.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # Cached so the (relatively slow) model load only happens once per process.
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(texts, convert_to_numpy=True).tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
