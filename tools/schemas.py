from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""

    expression: str = Field(..., description="A math expression, e.g. '2 + 3 * 4'.")

class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""

    query: str = Field(..., description="The search query.")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results to return.")

class SaveNoteInput(BaseModel):
    """Input schema for the save_note tool."""

    title: str = Field(..., description="Short title for the note.")
    content: str = Field(..., description="The note's body text.")
    tags: list[str] = Field(default_factory=list, description="Optional tags for later filtering.")

class DraftDocumentInput(BaseModel):
    """Input schema for the draft_document tool."""

    topic: str = Field(..., description="What the document is about.")
    content: str = Field(..., description="The fully drafted document text.")
    format: str = Field(..., description="Document format, e.g. 'email', 'report', 'readme'.")
    tone: str = Field(..., description="Desired tone, e.g. 'formal', 'casual', 'persuasive'.")

class GetDraftInput(BaseModel):
    """Input schema for the get_draft tool."""

    doc_id: str = Field(..., description="The id of the draft to retrieve.")

class ReviseDocumentInput(BaseModel):
    """Input schema for the revise_document tool."""

    doc_id: str = Field(..., description="The id of the draft to revise.")
    new_content: str = Field(..., description="The full revised document text.")
    instruction: str = Field(..., description="The edit instruction that was applied.")

class CreateTaskInput(BaseModel):
    """Input schema for the create_task tool."""

    title: str = Field(..., description="Short title for the task.")
    steps: list[str] = Field(..., min_length=1, description="Ordered actionable steps.")

class GetTaskInput(BaseModel):
    """Input schema for the get_task tool."""

    task_id: str = Field(..., description="The id of the task to retrieve.")

class MarkStepDoneInput(BaseModel):
    """Input schema for the mark_step_done tool."""

    task_id: str = Field(..., description="The id of the task.")
    step_index: int = Field(..., ge=0, description="Zero-based index of the step to mark done.")
