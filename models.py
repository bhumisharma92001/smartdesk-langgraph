"""Application-scoped language-model clients."""
import os
from functools import lru_cache

from langchain_openrouter.chat_models import ChatOpenRouter

_REQUEST_TIMEOUT_MS = 45_000

_REASONING_MODEL = os.getenv(
    "SMARTDESK_MODEL", "openrouter/free"
)
_FAST_MODEL = os.getenv("SMARTDESK_FAST_MODEL", _REASONING_MODEL)


@lru_cache(maxsize=4)
def reasoning_model(temperature: float = 0) -> ChatOpenRouter:
    """Reuse the higher-quality model for specialist work."""
    return ChatOpenRouter(
        model=_REASONING_MODEL, temperature=temperature, max_tokens=2048,
        timeout=_REQUEST_TIMEOUT_MS, max_retries=0,
    )


@lru_cache(maxsize=1)
def fast_model() -> ChatOpenRouter:
    """Reuse the low-latency model for orchestration and memory maintenance."""
    return ChatOpenRouter(
        model=_FAST_MODEL, temperature=0, max_tokens=512,
        timeout=_REQUEST_TIMEOUT_MS, max_retries=0,
    )
