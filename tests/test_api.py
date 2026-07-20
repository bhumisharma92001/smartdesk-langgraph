"""Offline tests for API background maintenance."""
import threading
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import api


class _Graph:
    def __init__(self) -> None:
        self.updated = None

    def get_state(self, config):
        return SimpleNamespace(values={"messages": []})

    def update_state(self, config, patch):
        self.updated = (config, patch)


def test_background_summary_updates_checkpoint(monkeypatch):
    graph = _Graph()
    monkeypatch.setattr(api, "summarize_history_node", lambda state: {"summary": "facts"})

    api._maybe_summarize(graph, {"configurable": {"thread_id": "t1"}}, threading.Lock())

    assert graph.updated[1] == {"summary": "facts"}


def test_public_history_keeps_only_final_assistant_message_per_user_turn():
    messages = [
        HumanMessage(content="research and write"),
        AIMessage(content="intermediate research"),
        ToolMessage(content="internal", tool_call_id="call-1"),
        AIMessage(content="final draft"),
        HumanMessage(content="thanks"),
        AIMessage(content="you are welcome"),
    ]

    assert api._public_history(messages) == [
        {"role": "user", "content": "research and write"},
        {"role": "assistant", "content": "final draft"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "you are welcome"},
    ]
