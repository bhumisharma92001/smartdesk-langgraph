import operator
from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def add_unique(left: list[str], right: list[str]) -> list[str]:
    """Reducer: append items from ``right`` onto ``left``, skipping exact duplicates.

    Used for the ``memories`` channel so that cross-session memory
    entries accumulate over the life of a thread without the same
    string being stored more than once.

    Args:
        left: The existing accumulated list (current channel value).
        right: The new items a node is writing to the channel.

    Returns:
        ``left`` with any not-already-present items from ``right`` appended.
    """
    existing = set(left)
    return left + [item for item in right if item not in existing]


class GlobalState(TypedDict):
    """Shared state threaded through every node in the SmartDesk graph."""

    messages: Annotated[list, add_messages]
    user_id: str
    active_agent: Optional[str]
    task_queue: list[dict[str, Any]]
    completed_tasks: Annotated[list, operator.add]
    memories: Annotated[list[str], add_unique]
    summary: str
    error: Optional[str]