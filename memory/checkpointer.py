"""SQLite-backed thread persistence for LangGraph state."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_DB_PATH = "smartdesk_checkpoints.sqlite"

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")


@contextmanager
def build_sqlite_checkpointer(db_path: str = DEFAULT_DB_PATH) -> Iterator[SqliteSaver]:
    """Yield a checkpointer and close its connection on exit."""
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        yield checkpointer


def make_thread_config(thread_id: str, recursion_limit: int = 20) -> dict:
    """Build the config required by every graph invocation."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
