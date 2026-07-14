from pydantic import BaseModel, Field
from typing import Optional

class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""

    expression: str = Field(..., description="A math expression, e.g. '2 + 3 * 4'.")

class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""

    query: str = Field(..., description="The search query.")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results to return.")

class FetchPageInput(BaseModel):
    url: str = Field(..., description="The URL to fetch and extract text from.")
    max_chars: int = Field(3000, ge=500, le=10000, description="Max characters of extracted text to return.")

class SaveNoteInput(BaseModel):
    """Input schema for the save_note tool."""

    title: str = Field(..., description="Short title for the note.")
    content: str = Field(..., description="The note's body text.")
    tags: list[str] = Field(default_factory=list, description="Optional tags for later filtering.")

class ListNotesInput(BaseModel):
    """Input schema for the list_notes tool."""

    tag_filter: Optional[str] = Field(None, description="If set, only return notes with this tag.")