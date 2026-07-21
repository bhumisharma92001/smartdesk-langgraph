"""Single cached model factory."""
import os
from functools import lru_cache
from langchain_openai import ChatOpenAI


@lru_cache(maxsize=1)
def model() -> ChatOpenAI:
    """Return one process-stable model; environment changes require restart."""
    name = os.getenv("SMARTDESK_MODEL", "openrouter/free")
    return ChatOpenAI(model=name, api_key=os.environ["OPENROUTER_API_KEY"],
                      base_url="https://openrouter.ai/api/v1", temperature=0,
                      timeout=30, max_retries=0, max_tokens=2048)
