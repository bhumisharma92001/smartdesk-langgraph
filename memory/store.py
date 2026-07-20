from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
import threading
import uuid
from typing import Iterator, TypedDict, cast

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore
from state import TaskRecord


class NoteRecord(TypedDict):
    note_id: str
    title: str
    content: str
    tags: list[str]


class RevisionRecord(TypedDict):
    instruction: str
    previous_content: str


class DraftRecord(TypedDict):
    doc_id: str
    topic: str
    content: str
    format: str
    tone: str
    revisions: list[RevisionRecord]


class SmartDeskStore:
    """Per-user store for notes, tasks, and drafted documents."""

    def __init__(self, backend: BaseStore | None = None) -> None:
        """Wrap the given BaseStore, or a fresh InMemoryStore if none given."""
        self._backend = backend or InMemoryStore()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def save_note(self, user_id: str, title: str, content: str, tags: list[str] | None = None) -> str:
        """Save a note for user_id and return its generated note_id."""
        note_id = str(uuid.uuid4())
        self._backend.put(("notes", user_id), note_id,
                           {"note_id": note_id, "title": title, "content": content, "tags": tags or []})
        return note_id

    def list_notes(self, user_id: str, tag_filter: str | None = None) -> list[NoteRecord]:
        """List user_id's notes, optionally filtered to those containing tag_filter."""
        notes = [cast(NoteRecord, item.value) for item in self._backend.search(("notes", user_id))]
        return [n for n in notes if tag_filter in n["tags"]] if tag_filter else notes

    def create_task(self, user_id: str, title: str, steps: list[str]) -> str:
        """Create a task with ordered steps and return its generated task_id."""
        task_id = str(uuid.uuid4())
        task = {"task_id": task_id, "title": title, "status": "in_progress",
                "steps": [{"description": s, "done": False} for s in steps]}
        self._backend.put(("tasks", user_id), task_id, task)
        return task_id

    def get_task(self, user_id: str, task_id: str) -> TaskRecord:
        """Fetch a task by id; raises KeyError if it doesn't exist for this user."""
        item = self._backend.get(("tasks", user_id), task_id)
        if item is None:
            raise KeyError(f"No task {task_id!r} for user {user_id!r}")
        return cast(TaskRecord, item.value)

    def mark_step_done(self, user_id: str, task_id: str, step_index: int) -> TaskRecord:
        """Complete the next ordered step and update task status."""
        with self._lock_for(f"task:{user_id}:{task_id}"):
            task = self.get_task(user_id, task_id)
            steps = task["steps"]
            if not 0 <= step_index < len(steps):
                raise IndexError(f"step_index {step_index} out of range (has {len(steps)} steps)")

            next_unfinished = next((i for i, s in enumerate(steps) if not s["done"]), None)
            if next_unfinished is None:
                raise ValueError(f"All steps already completed for task {task_id!r}")
            if step_index != next_unfinished:
                raise ValueError(
                    f"Steps must be completed in order: expected step_index "
                    f"{next_unfinished} (the next unfinished step), got {step_index}"
                )

            steps[step_index]["done"] = True
            task["status"] = "completed" if all(s["done"] for s in steps) else "in_progress"
            self._backend.put(("tasks", user_id), task_id, task)
            return task

    def save_draft(self, user_id: str, topic: str, content: str, format: str, tone: str) -> str:
        """Save a new drafted document and return its generated doc_id."""
        doc_id = str(uuid.uuid4())
        draft = {
            "doc_id": doc_id, "topic": topic, "content": content,
            "format": format, "tone": tone, "revisions": [],
        }
        self._backend.put(("drafts", user_id), doc_id, draft)
        return doc_id

    def get_draft(self, user_id: str, doc_id: str) -> DraftRecord:
        """Fetch a draft by id; raises KeyError if it doesn't exist for this user."""
        item = self._backend.get(("drafts", user_id), doc_id)
        if item is None:
            raise KeyError(f"No draft {doc_id!r} for user {user_id!r}")
        return cast(DraftRecord, item.value)

    def revise_draft(self, user_id: str, doc_id: str, new_content: str, instruction: str) -> DraftRecord:
        """Replace a draft's content, keeping a log of the instruction that produced the revision."""
        with self._lock_for(f"draft:{user_id}:{doc_id}"):
            draft = self.get_draft(user_id, doc_id)
            draft["revisions"].append({"instruction": instruction, "previous_content": draft["content"]})
            draft["content"] = new_content
            self._backend.put(("drafts", user_id), doc_id, draft)
            return draft


@lru_cache(maxsize=16)
def get_smartdesk_store(backend: BaseStore) -> SmartDeskStore:
    """Return one lightweight domain adapter per LangGraph store instance."""
    return SmartDeskStore(backend)


@contextmanager
def build_persistent_store(
    db_path: str = "smartdesk_memory.sqlite",
) -> Iterator[BaseStore]:
    """Yield persistent storage for notes, tasks, and drafts."""
    with SqliteStore.from_conn_string(db_path) as store:
        store.setup()
        yield store
