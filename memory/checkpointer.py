"""Layer 2: SQLite-backed thread memory."""
import security  # noqa: F401 - set strict deserialization before LangGraph imports

from contextlib import contextmanager
from hashlib import sha256

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from observability import log
logger = log("runtime")


def embed(texts: list[str]) -> list[list[float]]:
    """Load the embedding stack lazily when the semantic store needs vectors."""
    from memory.embeddings import embed as encode
    return encode(texts)


def user_id(username: str) -> str:
    """Derive the same private user identifier from a unique username."""
    return sha256(username.strip().lower().encode()).hexdigest()[:16]


def thread_config(owner_id: str, thread: str = "main") -> dict:
    """Build a user-owned isolated thread with the required recursion limit."""
    return {"configurable": {"thread_id": f"{owner_id}:{thread}"},
            "recursion_limit": 20}


@contextmanager
def runtime(database: str = "smartdesk.sqlite", memory_db: str = "smartdesk_memory.sqlite"):
    """Yield persistent semantic memory and thread checkpoints."""
    logger.info("opening checkpoint and memory stores")
    index = {"dims": 384, "embed": embed, "fields": ["text"]} #personal fact semantic search
    with SqliteSaver.from_conn_string(database) as checkpointer:
        with SqliteStore.from_conn_string(memory_db, index=index) as store:
            logger.info("initializing memory schema")
            store.setup()
            logger.info("runtime ready")
            yield store, checkpointer
