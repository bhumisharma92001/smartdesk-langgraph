"""Tests for shared agent retry and tool-error normalization."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.common import _unrecovered_tool_error, compact_history


def test_final_answer_recovers_an_earlier_tool_error():
    messages = [
        ToolMessage(content="404", tool_call_id="1", name="fetch_page", status="error"),
        AIMessage(content="I used the other search results instead."),
    ]
    assert _unrecovered_tool_error(messages) is None


def test_terminal_tool_error_is_reported():
    messages = [
        ToolMessage(content="404", tool_call_id="1", name="fetch_page", status="error")
    ]
    assert _unrecovered_tool_error(messages) == "fetch_page: 404"


def test_compact_history_removes_internal_tool_trace():
    messages = [
        HumanMessage(content="research this"),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "1"}]),
        ToolMessage(content="large search payload", tool_call_id="1", name="web_search"),
        AIMessage(content="compact final answer"),
    ]
    assert [message.content for message in compact_history(messages)] == [
        "research this", "compact final answer"
    ]
