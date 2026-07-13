from pydantic import BaseModel, Field


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