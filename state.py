"""Shared LangGraph state."""
import operator
from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def merge_error(left: str | None, right: str | None) -> str | None:
    """Combine failures produced by concurrent agent branches."""
    if right is None: return None
    return f"{left}; {right}" if left else right


def merge_tasks(left: list[dict], right: list[dict]) -> list[dict]:
    """Upsert pending tasks and remove tasks completed by TaskAgent."""
    tasks = {task["task_id"]: task for task in left}
    for task in right:
        if task.get("status") == "completed": tasks.pop(task["task_id"], None)
        else: tasks[task["task_id"]] = task
    return list(tasks.values())


def reset_or_add(left: list[dict], right: list[dict] | None) -> list[dict]:
    """Append concurrent turn data, or explicitly clear transient data."""
    return [] if right is None else left + right


class GlobalState(TypedDict, total=False):
    """State shared by the supervisor and all sub-agents."""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    active_agent: str | None
    turn_id: str
    task_queue: Annotated[list[dict], merge_tasks]
    completed_tasks: Annotated[list[dict], operator.add]
    current_document: dict | None
    last_note_list: list[dict]
    agent_outputs: Annotated[list[dict], reset_or_add]
    monitor_log: Annotated[list[dict], reset_or_add]
    memories: list[str]
    new_memories: list[str]
    summary: str
    error: Annotated[str | None, merge_error]
