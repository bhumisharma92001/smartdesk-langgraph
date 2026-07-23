"""Writer tools."""
import json
from typing import Annotated
from uuid import uuid4
from langchain_core.tools import ToolException, tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from tools.schemas import DraftDocumentInput, ReviseDocumentInput
from models import model


def generate_document(prompt: str) -> str:
    """Generate a complete document or fail before persisting truncated output."""
    response = model().bind(max_tokens=4096).invoke(prompt)
    finish_reason = response.response_metadata.get("finish_reason")
    if finish_reason in {"length", "max_tokens"}:
        raise ValueError("model output reached the document token limit")
    content = str(response.content).strip()
    if not content:
        raise ValueError("model returned an empty document")
    return content


@tool("draft_document", args_schema=DraftDocumentInput, response_format="content_and_artifact")
def draft_document(topic: str, format: str, tone: str,
                   user_id: Annotated[str, InjectedState("user_id")],
                   store: Annotated[BaseStore, InjectedStore()]) -> tuple[str, dict]:
    """Create and persist a concise document draft in the requested format and tone."""
    try:
        doc_id = str(uuid4())
        prompt = f"Write a {tone} {format} about {topic}. Return only the document."
        content = generate_document(prompt)
        doc = {"doc_id": doc_id, "topic": topic, "format": format, "tone": tone,
               "content": content}
        store.put(("documents", user_id), doc_id, doc, index=False)
        return json.dumps(doc), doc
    except Exception as exc:
        raise ToolException(f"Could not draft document: {exc}") from exc


@tool("revise_document", args_schema=ReviseDocumentInput, response_format="content_and_artifact")
def revise_document(doc_id: str, instruction: str,
                    user_id: Annotated[str, InjectedState("user_id")],
                    store: Annotated[BaseStore, InjectedStore()]) -> tuple[str, dict]:
    """Apply an edit instruction to an existing persisted draft."""
    try:
        namespace = ("documents", user_id)
        item = store.get(namespace, doc_id)
        if item is None:
            raise KeyError(doc_id)
        doc = dict(item.value)
        prompt = f"Revise this document as instructed.\nInstruction: {instruction}\n\n{doc['content']}"
        doc["content"] = generate_document(prompt)
        store.put(namespace, item.key, doc, index=False)
        return json.dumps(doc), doc
    except Exception as exc:
        raise ToolException(f"Could not revise document: {exc}") from exc
