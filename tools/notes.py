"""Note tools."""
import re
from typing import Annotated
from uuid import uuid4
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from tools.schemas import ListNotesInput, SaveNoteInput


@tool("save_note", args_schema=SaveNoteInput)
def save_note(title: str, content: str, tags: list[str],
              messages: Annotated[list, InjectedState("messages")],
              user_id: Annotated[str, InjectedState("user_id")],
              store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Persist a structured note when findings should survive the current turn."""
    try:
        latest = next((str(getattr(message, "content", "")) for message in reversed(messages)
                       if getattr(message, "type", "") == "human"), "")
        if not re.search(r"\b(note|save|remember|record)\b", latest, re.I):
            raise ValueError("the current user request did not authorize saving a note")
        existing = [item.value for item in store.search(("notes", user_id))]
        normalized = re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()
        duplicate = next((note for note in existing
                          if re.sub(r"[^a-z0-9]+", " ",
                                    str(note.get("title", "")).casefold()).strip() == normalized), None)
        if duplicate: return str(duplicate["id"])
        note_id = str(uuid4())
        store.put(("notes", user_id), note_id,
                  {"id": note_id, "title": title, "content": content, "tags": tags}, index=False)
        return note_id
    except Exception as exc:
        raise ToolException(f"Could not save note: {exc}") from exc

@tool("list_notes", args_schema=ListNotesInput)
def list_notes(tag_filter: str | None, note_id: str | None,
               user_id: Annotated[str, InjectedState("user_id")],
               store: Annotated[BaseStore, InjectedStore()]) -> list[dict]:
    """Retrieve a note by exact ID, or list saved notes with an optional tag filter."""
    try:
        namespace = ("notes", user_id)
        if note_id:
            item = store.get(namespace, note_id)
            return [item.value] if item else []
        notes = [item.value for item in store.search(namespace)]
        return [note for note in notes if not tag_filter or tag_filter in note["tags"]]
    except Exception as exc:
        raise ToolException(f"Could not list notes: {exc}") from exc
