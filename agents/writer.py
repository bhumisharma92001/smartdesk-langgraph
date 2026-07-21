"""WriterAgent subgraph."""
import json
from langchain_core.messages import ToolMessage
from langgraph.store.base import BaseStore
from agents.base import run
from state import GlobalState
from tools.documents import draft_document, revise_document
from tools.notes import list_notes
from tools.calculator import calculator


def run_writer(state: GlobalState, store: BaseStore) -> dict:
    """Draft or revise a document using prior research messages."""
    prompt = ("You are WriterAgent. Draft clear documents and preserve useful research citations. "
              "When revising the current document, use its exact doc_id from context. After a "
              "document tool succeeds, state that it is ready; the supervisor will display the "
              "canonical persisted content.")
    patch = run(state, store, [draft_document, revise_document, list_notes, calculator], prompt,
                "writer")
    results = patch.pop("tool_results", [])
    for message in reversed(results):
        if not isinstance(message, ToolMessage) or message.name not in {
                "draft_document", "revise_document"}:
            continue
        document = getattr(message, "artifact", None)
        if not isinstance(document, dict):
            try: document = json.loads(str(message.content))
            except (json.JSONDecodeError, TypeError): continue
        if "doc_id" in document and "content" in document:
            patch["current_document"] = {**document, "turn_id": state["turn_id"]}
            break
    return patch
