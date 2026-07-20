"""Offline contract tests for required tool schemas."""
import pytest
from pydantic import BaseModel

from tools.calculator import calculator
from tools.notes import save_note
from tools.search import web_search
from tools.tasks import create_task, mark_step_done
from tools.writer import draft_document, revise_document


@pytest.mark.parametrize(
    "tool",
    [web_search, save_note, create_task, mark_step_done,
     draft_document, revise_document, calculator],
)
def test_required_tools_use_pydantic_schemas_and_docstrings(tool):
    assert issubclass(tool.args_schema, BaseModel)
    assert tool.description and len(tool.description.split()) >= 6
