"""Validated contracts exchanged between agents and supervisor."""
import re
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl, TypeAdapter, ValidationError

URL = TypeAdapter(HttpUrl)


def cited_urls(text: str) -> list[str]:
    """Return only valid URLs explicitly cited in final agent content."""
    urls = []
    for raw in re.findall(r"https?://[^\s\])}'\"]+", text):
        try: urls.append(str(URL.validate_python(raw.rstrip(".,;:"))))
        except ValidationError: continue
    return sorted(set(urls))


class AgentReport(BaseModel):
    """Machine-checkable specialist result and execution evidence."""
    agent: Literal["research", "task", "writer"]
    turn_id: str
    status: Literal["success", "partial", "failed"]
    content: str
    sources: list[HttpUrl] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
