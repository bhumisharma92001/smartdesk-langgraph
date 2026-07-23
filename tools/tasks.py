"""Task queue tools."""
import json
import re
from typing import Annotated
from uuid import uuid4
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from tools.schemas import CreateTaskInput, MarkStepDoneInput

UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.I)


def render_tasks(tasks: list[dict]) -> str:
    """Format checkpointed tasks without adding another public tool."""
    if not tasks: return "No matching task found."
    return "\n\n".join(f'{task["title"]} ({task["task_id"]}) - {task["status"]}\n' +
        "\n".join(f'{i}. {step["text"]} - {"done" if step["done"] else "pending"}'
                   for i, step in enumerate(task["steps"])) for task in tasks)


@tool("create_task", args_schema=CreateTaskInput, response_format="content_and_artifact")
def create_task(title: str, steps: list[str],
                messages: Annotated[list, InjectedState("messages")],
                user_id: Annotated[str, InjectedState("user_id")],
                store: Annotated[BaseStore, InjectedStore()]) -> tuple[str, dict]:
    """Create a task with ordered sub-steps for a multi-step goal."""
    try:
        visible = "\n".join(str(getattr(message, "content", "")) for message in messages)
        invented = {match.group(0) for text in [title, *steps] for match in UUID.finditer(text)
                    if match.group(0).casefold() not in visible.casefold()}
        if invented:
            raise ValueError("task text contains an identifier absent from visible context")
        task_id = str(uuid4())
        task = {"task_id": task_id, "title": title, "status": "in_progress",
                "steps": [{"text": step, "done": False} for step in steps]}
        store.put(("tasks", user_id), task_id, task, index=False); return json.dumps(task), task
    except Exception as exc:
        raise ToolException(f"Could not create task: {exc}") from exc


@tool("mark_step_done", args_schema=MarkStepDoneInput, response_format="content_and_artifact")
def mark_step_done(task_id: str, step_index: int,
                   messages: Annotated[list, InjectedState("messages")],
                   user_id: Annotated[str, InjectedState("user_id")],
                   store: Annotated[BaseStore, InjectedStore()]) -> tuple[str, dict]:
    """Mark one specific task step complete after the user reports completion."""
    try:
        latest = next((str(getattr(message, "content", "")) for message in reversed(messages)
                       if getattr(message, "type", "") == "human"), "")
        if not re.search(r"\b(mark|done|complete|completed|finish|finished)\b", latest, re.I):
            raise ValueError("the current user request did not authorize completing a task step")
        item = store.get(("tasks", user_id), task_id)
        if item is None: raise KeyError(task_id)
        task = item.value
        if step_index < 0 or step_index >= len(task["steps"]):
            raise ValueError(f"step_index {step_index} out of range")
        task["steps"][step_index]["done"] = True
        task["status"] = "completed" if all(s["done"] for s in task["steps"]) else "in_progress"
        store.put(("tasks", user_id), task_id, task, index=False); return json.dumps(task), task
    except Exception as exc:
        raise ToolException(f"Could not update task: {exc}") from exc
