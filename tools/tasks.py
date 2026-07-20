import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore

from custom_exception.exceptions import ToolExecutionError
from memory.store import get_smartdesk_store
from tools.schemas import CreateTaskInput, GetTaskInput, MarkStepDoneInput


def _task_was_loaded(messages: list, task_id: str) -> bool:
    """Return whether an earlier successful get_task loaded this task."""
    for message in reversed(messages):
        if not isinstance(message, ToolMessage) or message.name != "get_task":
            continue
        if message.status == "error":
            continue
        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue
        if isinstance(content, dict) and content.get("task_id") == task_id:
            return True
    return False


@tool("create_task", args_schema=CreateTaskInput)
def create_task(
    title: str,
    steps: list[str],
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> dict:
    """Create and persist an ordered checklist for a new high-level goal."""
    try:
        task_store = get_smartdesk_store(store)
        task_id = task_store.create_task(user_id, title, steps)
        return task_store.get_task(user_id, task_id)
    except Exception as exc:
        raise ToolExecutionError(f"Could not create task {title!r}: {exc}") from exc


@tool("get_task", args_schema=GetTaskInput)
def get_task(
    task_id: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> dict:
    """Read a persisted task before choosing the next step to complete."""
    try:
        return get_smartdesk_store(store).get_task(user_id, task_id)
    except KeyError as exc:
        raise ToolExecutionError(f"Could not get task: {exc}") from exc
    except Exception as exc:
        raise ToolExecutionError(f"Could not retrieve task {task_id!r}: {exc}") from exc


@tool("mark_step_done", args_schema=MarkStepDoneInput)
def mark_step_done(
    task_id: str,
    step_index: int,
    user_id: Annotated[str, InjectedState("user_id")],
    messages: Annotated[list, InjectedState("messages")],
    store: Annotated[BaseStore, InjectedStore()],
) -> dict:
    """Complete the next ordered step after get_task has confirmed its index."""
    try:
        if not _task_was_loaded(messages, task_id):
            raise ValueError("Call get_task successfully before mark_step_done")
        return get_smartdesk_store(store).mark_step_done(user_id, task_id, step_index)
    except (KeyError, IndexError, ValueError) as exc:
        raise ToolExecutionError(f"Could not mark step done: {exc}") from exc
    except Exception as exc:
        raise ToolExecutionError(f"Could not update task {task_id!r}: {exc}") from exc
