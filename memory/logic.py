"""Minimal in-context and cross-session memory nodes."""
from uuid import uuid4
from langgraph.store.base import BaseStore
from memory.context import trim_history
from observability import log
from state import GlobalState
logger = log("memory")


def load_memories(state: GlobalState, store: BaseStore) -> dict:
    """Retrieve the three user memories most relevant to the latest message."""
    patch: dict = {"memories": [], "error": None}
    if state.get("active_agent") != "chat":
        query = str(state["messages"][-1].content).lower()
        namespace = ("memories", state["user_id"])
        candidates = store.search(namespace, limit=1)
        logger.info("semantic search started" if candidates else "semantic search skipped; store empty")
        items = store.search(namespace, query=query, limit=3) if candidates else []
        patch["memories"] = [item.value["text"] for item in items]
        logger.info("retrieved=%d", len(items))
    else: logger.info("semantic search skipped for chat")
    trimmed = trim_history(state); patch.update(trimmed)
    if trimmed: logger.info("history summarized and trimmed")
    return patch


def save_memories(state: GlobalState, store: BaseStore) -> dict:
    """Save explicit user facts or preferences after a successful run."""
    facts = state.get("new_memories", [])
    if not facts:
        logger.info("saved=0"); return {}
    namespace = ("memories", state["user_id"])
    existing = {item.value["text"] for item in store.search(namespace)}
    for fact in facts:
        if fact not in existing: store.put(namespace, str(uuid4()), {"text": fact})
    logger.info("facts=%d new=%d", len(facts), len(set(facts) - existing))
    return {}
