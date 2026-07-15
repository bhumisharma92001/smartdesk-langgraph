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

    def save_draft(self, user_id: str, topic: str, content: str, format: str, tone: str) -> str:
        """Save a new drafted document and return its generated doc_id."""
        doc_id = str(uuid.uuid4())
        draft = {
            "doc_id": doc_id, "topic": topic, "content": content,
            "format": format, "tone": tone, "revisions": [],
        }
        self._backend.put(("drafts", user_id), doc_id, draft)
        return doc_id
 
    def get_draft(self, user_id: str, doc_id: str) -> dict[str, Any]:
        """Fetch a draft by id; raises KeyError if it doesn't exist for this user."""
        item = self._backend.get(("drafts", user_id), doc_id)
        if item is None:
            raise KeyError(f"No draft {doc_id!r} for user {user_id!r}")
        return item.value
 
    def revise_draft(self, user_id: str, doc_id: str, new_content: str, instruction: str) -> dict[str, Any]:
        """Replace a draft's content, keeping a log of the instruction that produced the revision."""
        draft = self.get_draft(user_id, doc_id)
        draft["revisions"].append({"instruction": instruction, "previous_content": draft["content"]})
        draft["content"] = new_content
        self._backend.put(("drafts", user_id), doc_id, draft)
        return draft