"""Pydantic schemas for the exact nine required tools."""
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Web search input."""
    query: str
    max_results: int = Field(5, ge=1, le=10)


class FetchPageInput(BaseModel):
    """Page retrieval input."""
    url: str


class SaveNoteInput(BaseModel):
    """Note creation input."""
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)


class ListNotesInput(BaseModel):
    """Note filter input."""
    tag_filter: str | None = None


class CreateTaskInput(BaseModel):
    """Task creation input."""
    title: str
    steps: list[str] = Field(min_length=1)


class MarkStepDoneInput(BaseModel):
    """Task update input."""
    task_id: str
    step_index: int = Field(ge=0)


class DraftDocumentInput(BaseModel):
    """Document draft input."""
    topic: str
    format: str
    tone: str


class ReviseDocumentInput(BaseModel):
    """Document revision input."""
    doc_id: str
    instruction: str


class CalculatorInput(BaseModel):
    """Calculator input."""
    expression: str
