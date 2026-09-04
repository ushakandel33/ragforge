"""
Per-client rate limiting using slowapi (a FastAPI-friendly wrapper around
the `limits` library). Limits are keyed by client IP by default.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.config import settings

limiter = Limiter(key_func=get_remote_address)

CHAT_RATE_LIMIT = f"{settings.rate_limit_per_minute}/minute"
