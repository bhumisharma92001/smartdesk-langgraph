"""TaskAgent subgraph."""
import json
from langgraph.store.base import BaseStore
from agents.base import run
from state import GlobalState
from tools.notes import save_note, list_notes
from tools.calculator import calculator
from tools.tasks import create_task, mark_step_done


def run_task(state: GlobalState, store: BaseStore) -> dict:
    """Create, read, or advance an ordered task using persisted state as source of truth."""
    prompt = (
        "You are TaskAgent. Active tasks (source of truth, persisted): "
        f"{state.get('task_queue', [])}\nCompleted tasks: {state.get('completed_tasks', [])}\n"
        "Never invent a task_id — only use IDs shown above. "
        "For a new goal, call create_task. For completion requests, call mark_step_done using "
        "the exact persisted task_id and the correct step_index. "
        "If asked to show or list tasks, answer directly from the tasks shown above — "
        "no tool call is needed for reads."
    )
    patch = run(state, store, [create_task, mark_step_done, save_note, list_notes, calculator],
                prompt, "task")
    results = patch.pop("tool_results", [])
    tasks, invalid = [], False
    for message in results:
        if message.name not in {"create_task", "mark_step_done"}: continue
        task = getattr(message, "artifact", None)
        if not isinstance(task, dict):
            try: task = json.loads(str(message.content))
            except (json.JSONDecodeError, TypeError): invalid = True; continue
        if "task_id" in task: tasks.append(task)
    saved = {t["task_id"] for t in state.get("completed_tasks", [])}
    completed = [t for t in tasks if t.get("status") == "completed" and t["task_id"] not in saved]
    if tasks: patch["task_queue"] = tasks
    if completed: patch["completed_tasks"] = completed
    if invalid: patch["error"] = "Task tool returned an invalid state artifact"
    return patch