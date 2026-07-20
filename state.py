import operator
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TaskStep(TypedDict):
    description: str
    done: bool


class TaskRecord(TypedDict):
    task_id: str
    title: str
    status: str
    steps: list[TaskStep]


def update_task_queue(left: list[TaskRecord], right: list[TaskRecord]) -> list[TaskRecord]:
    """Upsert active tasks and remove completed ones."""
    queue = {t["task_id"]: t for t in left}
    for task in right:
        if task.get("status") == "completed":
            queue.pop(task.get("task_id"), None)
        else:
            queue[task["task_id"]] = task
    return list(queue.values())


def merge_errors(left: str | None, right: str | None) -> str | None:
    """Clear at turn start or combine concurrent branch failures.

    Successful nodes must omit the error key so a parallel success cannot
    erase another branch's failure.
    """
    if right is None:
        return None
    if left and left != right:
        return f"{left}; {right}"
    return right


class GlobalState(TypedDict):
    """Shared state threaded through every node in the SmartDesk graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    active_agent: str | None
    task_queue: Annotated[list[TaskRecord], update_task_queue]
    completed_tasks: Annotated[list[TaskRecord], operator.add]
    summary: str
    research_summary: str
    writer_output: str
    task_output: str
    routing_log: Annotated[list[dict], operator.add]
    error: Annotated[str | None, merge_errors]