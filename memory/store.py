from __future__ import annotations
import uuid
from typing import Any, Optional

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


class SmartDeskStore: #Instead of every tool interacting directly with LangGraph's BaseStore, they all talk to SmartDeskStore
    """Per-user store for notes and tasks."""

    def __init__(self, backend: Optional[BaseStore] = None) -> None:
        """Wrap the given BaseStore, or a fresh InMemoryStore if none given."""
        self._backend = backend or InMemoryStore()

    def save_note(self, user_id: str, title: str, content: str, tags: Optional[list[str]] = None) -> str:
        """Save a note for user_id and return its generated note_id."""
        note_id = str(uuid.uuid4())
        self._backend.put(("notes", user_id), note_id,
                           {"note_id": note_id, "title": title, "content": content, "tags": tags or []})
        return note_id

    def list_notes(self, user_id: str, tag_filter: Optional[str] = None) -> list[dict[str, Any]]:
        """List user_id's notes, optionally filtered to those containing tag_filter."""
        notes = [item.value for item in self._backend.search(("notes", user_id))]
        return [n for n in notes if tag_filter in n["tags"]] if tag_filter else notes

    def create_task(self, user_id: str, title: str, steps: list[str]) -> str:
        """Create a task with ordered steps and return its generated task_id."""
        task_id = str(uuid.uuid4())
        task = {"task_id": task_id, "title": title, "status": "in_progress",
                "steps": [{"description": s, "done": False} for s in steps]}
        self._backend.put(("tasks", user_id), task_id, task)
        return task_id

    def get_task(self, user_id: str, task_id: str) -> dict[str, Any]:
        """Fetch a task by id; raises KeyError if it doesn't exist for this user."""
        item = self._backend.get(("tasks", user_id), task_id)
        if item is None:
            raise KeyError(f"No task {task_id!r} for user {user_id!r}")
        return item.value

    def mark_step_done(self, user_id: str, task_id: str, step_index: int) -> dict[str, Any]:
        """Mark one step done, auto-completing the task once all steps are done."""
        task = self.get_task(user_id, task_id)
        steps = task["steps"]
        if not 0 <= step_index < len(steps):
            raise IndexError(f"step_index {step_index} out of range (has {len(steps)} steps)")
        steps[step_index]["done"] = True
        task["status"] = "completed" if all(s["done"] for s in steps) else "in_progress"
        self._backend.put(("tasks", user_id), task_id, task)
        return task