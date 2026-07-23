"""WriterAgent subgraph."""
import json
from langchain_core.messages import ToolMessage
from langgraph.store.base import BaseStore
from agents.base import build, run
from state import GlobalState
from tools.documents import draft_document, revise_document
from tools.notes import list_notes
from tools.calculator import calculator


def build_writer(store: BaseStore):
    """Compile WriterAgent and return its supervisor node."""
    prompt = ("You are WriterAgent. Draft clear documents and preserve useful research citations. "
              "When asked to create, draft, or write a document, you MUST call draft_document; "
              "never claim it is ready without a successful document tool. When revising the "
              "current document, use its exact doc_id. After a document tool succeeds, state that "
              "it is ready; the finalizer will display the canonical persisted content.")
    agent = build(store, [draft_document, revise_document, list_notes, calculator], prompt)

    def writer(state: GlobalState) -> dict:
        return run_writer(state, agent)

    return writer


def run_writer(state: GlobalState, agent) -> dict:
    """Draft or revise a document using prior research messages."""
    patch, results = run(state, agent, "writer")
    document_used = False
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
            document_used = True
            break
    if not document_used and not patch.get("error"):
        patch["error"] = "WriterAgent did not persist a document"
    return patch
