"""
Simple persistent response cache keyed on a hash of (query + context).
Avoids re-calling the LLM for repeated questions, cutting latency and
API usage. Backed by diskcache so it survives process restarts, unlike a
plain in-memory dict.
"""
import hashlib
import json
from diskcache import Cache
from app.config import settings

_cache = Cache("./data/response_cache")


def _make_key(query: str, context: str) -> str:
    raw = json.dumps({"query": query, "context": context}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(query: str, context: str = ""):
    return _cache.get(_make_key(query, context))


def set_cached(query: str, context: str, value):
    _cache.set(_make_key(query, context), value, expire=settings.cache_ttl_seconds)
