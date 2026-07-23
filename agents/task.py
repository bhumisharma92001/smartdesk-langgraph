"""TaskAgent subgraph."""
import json
from langgraph.store.base import BaseStore
from agents.base import build, run
from state import GlobalState
from tools.notes import save_note, list_notes
from tools.calculator import calculator
from tools.tasks import create_task, mark_step_done


def build_task(store: BaseStore):
    """Compile TaskAgent and return its supervisor node."""
    tools = [create_task, mark_step_done, save_note, list_notes, calculator]
    prompt = (
        "You are TaskAgent. Treat the active and completed tasks in the injected context as the "
        "source of truth, but always address the user's current message first. Only advance or "
        "reference active tasks when the current message asks about tasks. A note request must "
        "call save_note or list_notes and must never update a task. Never invent any task, note, "
        "document, or other identifier, including inside task step text; use an ID only when it "
        "appears in user input, injected context, or a tool result from this call. For a new goal, "
        "call create_task. For completion requests, call mark_step_done with the exact task_id "
        "and step_index. For an exact note ID, call list_notes with note_id."
    )
    agent = build(store, tools, prompt)

    def task(state: GlobalState) -> dict:
        return run_task(state, agent)

    return task


def run_task(state: GlobalState, agent) -> dict:
    """Create, read, or advance an ordered task using persisted state as source of truth."""
    patch, results = run(state, agent, "task")
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
