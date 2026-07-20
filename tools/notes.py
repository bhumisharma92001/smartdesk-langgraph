from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore

from custom_exception.exceptions import ToolExecutionError
from memory.store import get_smartdesk_store
from tools.schemas import SaveNoteInput


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
        return get_smartdesk_store(store).save_note(user_id, title, content, tags)
    except Exception as exc:
        raise ToolExecutionError(f"Could not save note {title!r}: {exc}") from exc
