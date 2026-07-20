"""Offline tests for supervisor routing, fan-out, handoff, and fallback behavior."""
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

import graph as graph_module
from state import GlobalState, merge_errors


def _state(text: str = "help") -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "user_id": "user-1",
        "active_agent": None,
        "task_queue": [],
        "completed_tasks": [],
        "summary": "",
        "research_summary": "",
        "writer_output": "",
        "task_output": "",
        "routing_log": [],
        "error": None,
    }


def _install_memory_stubs(monkeypatch) -> None:
    """Install graph stubs shared by orchestration tests."""


def test_route_sends_failures_to_error_handler():
    assert graph_module._route({"error": "boom", "active_agent": "task"}) == "error"


def test_summarization_is_not_on_graph_critical_path():
    compiled = graph_module.build_graph(store=InMemoryStore())
    assert "summarize" not in compiled.get_graph().nodes


def test_greeting_fast_path_skips_model(monkeypatch):
    def unexpected_model_call():
        raise AssertionError("greeting must not call the model")

    monkeypatch.setattr(graph_module, "fast_model", unexpected_model_call)
    patch = graph_module.supervisor_node(_state("hi"))
    assert patch["active_agent"] == "chat"
    assert patch["routing_log"][0]["workflow"] == "chat"


def test_active_task_continuations_route_to_task_without_model(monkeypatch):
    def unexpected_model_call():
        raise AssertionError("active-task continuation must not call router model")

    monkeypatch.setattr(graph_module, "fast_model", unexpected_model_call)
    active_task = {
        "task_id": "task-123",
        "title": "Prepare for interview",
        "status": "in_progress",
        "steps": [{"description": "Research company", "done": False}],
    }

    for reply in ("NEXT", "DONE", "continue", "mark done", "finished"):
        state = _state(reply)
        state["task_queue"] = [active_task]
        patch = graph_module.supervisor_node(state)
        assert patch["active_agent"] == "task"
        assert patch["error"] is None
        assert patch["routing_log"][0]["fallback"] is False


def test_task_continuation_without_active_task_still_uses_supervisor(monkeypatch):
    structured = SimpleNamespace(
        invoke=lambda _: {"parsed": graph_module.Route(workflow="chat"), "raw": None}
    )
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: structured)
    monkeypatch.setattr(graph_module, "fast_model", lambda: model)

    patch = graph_module.supervisor_node(_state("done"))
    assert patch["active_agent"] == "chat"


def test_chat_node_blocks_raw_tool_markup(monkeypatch):
    leaked = AIMessage(content=(
        "<tool_call>update_task<arg_key>task_id</arg_key>"
        "<arg_value>hallucinated-id</arg_value></tool_call>"
    ))
    monkeypatch.setattr(
        graph_module, "fast_model", lambda: SimpleNamespace(invoke=lambda _: leaked)
    )

    patch = graph_module.chat_node(_state("done"))
    content = patch["messages"][0].content
    assert "<tool_call>" not in content
    assert "arg_value" not in content
    assert "couldn't safely process" in content


def test_supervisor_routes_travel_plan_to_task(monkeypatch):
    structured = SimpleNamespace(
        invoke=lambda _: {"parsed": graph_module.Route(workflow="task"), "raw": None}
    )
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: structured)
    monkeypatch.setattr(graph_module, "fast_model", lambda: model)
    patch = graph_module.supervisor_node(_state("make a plan for trip to japan"))

    assert patch["active_agent"] == "task"
    assert patch["error"] is None
    assert patch["routing_log"][0] == {
        "input": "make a plan for trip to japan",
        "workflow": "task",
        "fallback": False,
    }


def test_supervisor_uses_structured_model_route(monkeypatch):
    structured = SimpleNamespace(
        invoke=lambda _: {
            "parsed": graph_module.Route(workflow="research_task_writer"), "raw": None,
        }
    )
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: structured)
    monkeypatch.setattr(graph_module, "fast_model", lambda: model)

    patch = graph_module.supervisor_node(_state("Research, plan, then write"))
    assert patch["active_agent"] == "research_task_writer"
    assert graph_module._after_research(patch) == "task"
    assert graph_module._after_task(patch) == "writer"


def test_supervisor_recovers_when_structured_model_returns_none(monkeypatch):
    structured = SimpleNamespace(invoke=lambda _: None)
    model = SimpleNamespace(with_structured_output=lambda *args, **kwargs: structured)
    monkeypatch.setattr(graph_module, "fast_model", lambda: model)

    patch = graph_module.supervisor_node(_state("Check the latest gold price for 2026"))

    assert patch["active_agent"] == "research"
    assert patch["error"] is None
    assert patch["routing_log"][0]["fallback"] is True


def test_router_fallback_does_not_treat_notes_as_web_research():
    assert graph_module._fallback_route("search my notes for current prices") == "notes"


def test_route_selectors_cover_notes_and_parallel_workflow():
    assert graph_module._route({"active_agent": "notes", "error": None}) == "notes"
    assert graph_module._route({"active_agent": "research_task", "error": None}) == "research_task"


def test_notes_node_lists_persisted_tagged_notes():
    store = InMemoryStore()
    from memory.store import SmartDeskStore

    SmartDeskStore(store).save_note(
        "user-1", "Checkpointing", "Verified summary", ["langgraph"]
    )
    patch = graph_module.notes_node(
        {
            "user_id": "user-1",
            "messages": [HumanMessage(content='List notes tagged "langgraph"')],
        },
        store,
    )
    assert "Verified summary" in patch["messages"][0].content


def test_parallel_research_and_task_fan_in(monkeypatch):
    calls: list[str] = []
    _install_memory_stubs(monkeypatch)
    monkeypatch.setattr(
        graph_module, "supervisor_node", lambda state: {"active_agent": "research_task", "error": None}
    )

    def research(state: GlobalState, store: BaseStore) -> dict:
        calls.append("research")
        return {"research_summary": "facts"}

    def task(state: GlobalState, store: BaseStore) -> dict:
        calls.append("task")
        return {
            "task_output": "steps",
            "task_queue": [{
                "task_id": "task-1",
                "title": "Recovery test",
                "status": "in_progress",
                "steps": [{"description": "Run test", "done": False}],
            }],
        }

    original_finalize = graph_module.finalize_node

    def finalize(state: GlobalState) -> dict:
        assert state["research_summary"] == "facts"
        assert state["task_output"] == "steps"
        calls.append("finalize")
        return original_finalize(state)

    monkeypatch.setattr(graph_module, "research_agent_node", research)
    monkeypatch.setattr(graph_module, "task_agent_node", task)
    monkeypatch.setattr(graph_module, "finalize_node", finalize)
    result = graph_module.build_graph(store=InMemoryStore()).invoke(_state())

    assert set(calls[:2]) == {"research", "task"}
    assert calls[-1] == "finalize"
    assert "facts" in result["messages"][-1].content
    assert "Task ID:** `task-1`" in result["messages"][-1].content


def test_research_writer_handoff_is_sequential(monkeypatch):
    calls: list[str] = []
    _install_memory_stubs(monkeypatch)
    monkeypatch.setattr(
        graph_module, "supervisor_node", lambda state: {"active_agent": "research_writer", "error": None}
    )

    def research(state: GlobalState, store: BaseStore) -> dict:
        calls.append("research")
        return {"research_summary": "verified source"}

    def writer(state: GlobalState, store: BaseStore) -> dict:
        assert state["research_summary"] == "verified source"
        calls.append("writer")
        return {"messages": [AIMessage(content="draft")], "writer_output": "draft"}

    monkeypatch.setattr(graph_module, "research_agent_node", research)
    monkeypatch.setattr(graph_module, "writer_agent_node", writer)
    result = graph_module.build_graph(store=InMemoryStore()).invoke(_state())

    assert calls == ["research", "writer"]
    assert result["writer_output"] == "draft"


def test_error_handler_returns_graceful_message():
    patch = graph_module.error_handler_node({"error": "tool unavailable"})
    assert "tool unavailable" not in patch["messages"][0].content
    assert "Please try again" in patch["messages"][0].content


def test_new_turn_clears_previous_error_and_agent_handoffs():
    assert graph_module.begin_turn_node({"error": "old failure"}) == {
        "error": None,
        "research_summary": "",
        "task_output": "",
        "writer_output": "",
    }


def test_concurrent_errors_are_preserved_and_turn_start_can_clear_them():
    combined = merge_errors("research failed", "task failed")
    assert combined == "research failed; task failed"
    assert merge_errors(combined, None) is None
