from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore

from custom_exception.exceptions import ToolExecutionError
from memory.store import SmartDeskStore
from tools.schemas import DraftDocumentInput, GetDraftInput, ReviseDocumentInput


@tool("draft_document", args_schema=DraftDocumentInput)
def draft_document(
    topic: str,
    content: str,
    format: str,
    tone: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Persist a drafted document and return its generated doc_id.

    Call this once you have composed the full document text yourself -
    this tool stores it, it does not generate content on your behalf.

    Raises:
        ToolExecutionError: If the draft could not be saved.
    """
    try:
        return SmartDeskStore(store).save_draft(user_id, topic, content, format, tone)
    except Exception as exc:
        raise ToolExecutionError(f"Could not save draft {topic!r}: {exc}") from exc


@tool("get_draft", args_schema=GetDraftInput)
def get_draft(
    doc_id: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> dict:
    """Retrieve an existing draft's content, format, and tone by id.

    Call this before revise_document whenever you don't already have the
    draft's current text in context - you need the existing content to
    produce a properly revised version rather than guessing at it.

    Raises:
        ToolExecutionError: If no draft exists with this id for this user.
    """
    try:
        return SmartDeskStore(store).get_draft(user_id, doc_id)
    except KeyError as exc:
        raise ToolExecutionError(f"Could not get draft: {exc}") from exc
    except Exception as exc:
        raise ToolExecutionError(f"Could not retrieve draft {doc_id!r}: {exc}") from exc


@tool("revise_document", args_schema=ReviseDocumentInput)
def revise_document(
    doc_id: str,
    new_content: str,
    instruction: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[BaseStore, InjectedStore()],
) -> dict:
    """Apply a revision to an existing draft and persist the new version.

    Use get_draft first to retrieve the current text, compose the revised
    text yourself, then call this to store it and log which instruction
    produced the change.

    Raises:
        ToolExecutionError: If the draft doesn't exist or can't be updated.
    """
    try:
        return SmartDeskStore(store).revise_draft(user_id, doc_id, new_content, instruction)
    except KeyError as exc:
        raise ToolExecutionError(f"Could not revise document: {exc}") from exc
    except Exception as exc:
        raise ToolExecutionError(f"Could not revise draft {doc_id!r}: {exc}") from exc