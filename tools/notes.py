from typing import Annotated, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore

from custom_exception.exceptions import ToolExecutionError
from memory.store import SmartDeskStore
from tools.schemas import ListNotesInput, SaveNoteInput


@tool("save_note", args_schema=SaveNoteInput)
def save_note(
    title: str,
    content: str,
    tags: list[str],
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Save a note to the user's persistent notes.

    Use this to record findings, facts, or context worth keeping for later
    in the task (e.g. research results, decisions made).

    Returns:
        The generated note_id.

    Raises:
        ToolExecutionError: If the note could not be saved.
    """
    try:
        return SmartDeskStore(store).save_note(user_id, title, content, tags)
    except Exception as exc:
        raise ToolExecutionError(f"Could not save note {title!r}: {exc}") from exc


@tool("list_notes", args_schema=ListNotesInput)
def list_notes(
    tag_filter: Optional[str],
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> list[dict]:
    """List the user's saved notes, optionally filtered by a single tag.

    Returns:
        A list of note dicts, each with note_id, title, content, and tags.

    Raises:
        ToolExecutionError: If notes could not be retrieved.
    """
    try:
        return SmartDeskStore(store).list_notes(user_id, tag_filter)
    except Exception as exc:
        raise ToolExecutionError(f"Could not list notes: {exc}") from exc