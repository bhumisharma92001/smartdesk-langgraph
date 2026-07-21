"""Fast regression tests for SmartDesk's critical local behavior."""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore
from types import SimpleNamespace

import finalizer
from finalizer import finalize
from graph import failed, join
from memory.checkpointer import runtime, thread_config
from memory.logic import load_memories
from monitoring import after_join
from state import merge_error, merge_tasks, reset_or_add
import supervisor
from supervisor import (classify_fallback, is_greeting, plan_workflow,
                        remove_unrequested_research, supervise)
from tools.calculator import calculator
from tools.notes import save_note
from tools.schemas import CreateTaskInput, WebSearchInput


def test_routing_selects_all_supported_workflows() -> None:
    """Route single, chained, and parallel requests without an LLM call."""
    assert classify_fallback("search current LangGraph news") == "research"
    assert classify_fallback("create a task checklist") == "task"
    assert classify_fallback("write an email") == "writer"
    assert classify_fallback("research this and write a report") == "research_writer"
    assert classify_fallback("research this and create a task") == "research_task"
    assert classify_fallback("research, plan tasks, and write a report") == "research_task_writer"
    assert classify_fallback("Create a task called Launch Website with steps: review content, "
                             "run tests, deploy to production.") == "task"
    assert classify_fallback("Create a website deployment task") == "task"


def test_research_guard_removes_unrequested_agent() -> None:
    """An over-eager LLM cannot turn website task steps into research."""
    text = "Create a launch task with review, test, and deploy steps"
    assert remove_unrequested_research("research_task", text) == "task"
    assert remove_unrequested_research("research_task_writer", text) == "task_writer"
    assert remove_unrequested_research("research_writer", "Write a report") == "writer"


def test_llm_planner_handles_keyword_free_task(monkeypatch) -> None:
    """Nuanced intent is handled by the LLM instead of regex vocabulary alone."""
    class Structured:
        def invoke(self, messages):
            return supervisor.Plan(workflow="task")

    class FakeModel:
        def with_structured_output(self, schema):
            return Structured()

    monkeypatch.setattr(supervisor, "model", lambda: FakeModel())
    assert plan_workflow("I need to get organized before the site goes live—break it down for me") == "task"


def test_llm_planner_falls_back_on_provider_failure(monkeypatch) -> None:
    """Provider failures preserve deterministic routing availability."""
    monkeypatch.setattr(supervisor, "model", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert plan_workflow("create a task checklist") == "task"


def test_note_save_is_idempotent_by_normalized_title() -> None:
    """Repeated note calls with cosmetic title changes return one canonical ID."""
    store = InMemoryStore()
    first = save_note.func("LangGraph Memory", "first", ["ai"], "u1", store)
    second = save_note.func(" langgraph-memory! ", "second", ["ai"], "u1", store)
    assert second == first
    assert len(store.search(("notes", "u1"))) == 1


def test_supervisor_resets_turn_transients() -> None:
    """Each new turn receives an ID and clears stale transient fields."""
    patch = supervise({"messages": [HumanMessage("hello!!!")]})
    assert is_greeting("helloooo")
    assert patch["active_agent"] == "chat"
    assert patch["turn_id"]
    assert patch["error"] is None
    assert patch["agent_outputs"] is None


def test_state_reducers_upsert_complete_and_clear() -> None:
    """Reducers merge parallel work and support explicit turn cleanup."""
    pending = {"task_id": "1", "status": "in_progress"}
    assert merge_tasks([], [pending]) == [pending]
    assert merge_tasks([pending], [{"task_id": "1", "status": "completed"}]) == []
    assert merge_error("old", None) is None
    assert merge_error("one", "two") == "one; two"
    assert reset_or_add([{"old": True}], None) == []


def test_tool_schemas_and_calculator() -> None:
    """Pydantic boundaries reject invalid inputs and arithmetic stays executable."""
    assert WebSearchInput(query="LangGraph", max_results=10).max_results == 10
    assert CreateTaskInput(title="Ship", steps=["test"]).steps == ["test"]
    assert calculator.invoke({"expression": "27 * 14"}) == 378.0


def test_memory_retrieves_top_three() -> None:
    """Memory loading asks the store for at most three semantic matches."""
    class Store:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def search(self, namespace, **kwargs):
            self.calls.append(kwargs)
            limit = kwargs.get("limit", 10)
            return [SimpleNamespace(value={"text": f"fact {index}"})
                    for index in range(limit)]

    store = Store()
    patch = load_memories({"messages": [HumanMessage("facts")], "user_id": "u1",
                           "active_agent": "task"}, store)  # type: ignore[arg-type]
    assert len(patch["memories"]) == 3
    assert any(call.get("limit") == 3 for call in store.calls)


def test_error_node_returns_graceful_message() -> None:
    """Unhandled workflow failures become a user-safe response."""
    patch = failed({"error": "secret internal failure"})
    assert isinstance(patch["messages"][0], AIMessage)
    assert "secret" not in patch["messages"][0].content


def test_chat_followup_returns_canonical_document(monkeypatch) -> None:
    """A 'share here' follow-up returns stored content without regenerating it."""
    monkeypatch.setattr(finalizer, "model", lambda: (_ for _ in ()).throw(
        AssertionError("LLM must not run")))
    state = {"messages": [HumanMessage("share here")], "active_agent": "chat",
             "turn_id": "t2", "current_document": {
                 "doc_id": "d1", "content": "EXACT SAVED REPORT", "turn_id": "t1"}}
    patch = finalize(state)
    assert patch["messages"][0].content == "EXACT SAVED REPORT"


def test_writer_turn_returns_persisted_document_not_preview(monkeypatch) -> None:
    """A new draft's canonical artifact overrides the agent's preview message."""
    monkeypatch.setattr(finalizer, "model", lambda: (_ for _ in ()).throw(
        AssertionError("LLM must not run")))
    state = {"messages": [HumanMessage("write a report")], "active_agent": "writer",
             "turn_id": "t1", "current_document": {
                 "doc_id": "d1", "content": "FULL CANONICAL REPORT", "turn_id": "t1"},
             "agent_outputs": [{"turn_id": "t1", "agent": "writer",
                                "content": "Would you like me to share it?"}],
             "monitor_log": [{"turn_id": "t1", "agent": "writer", "approved": True}]}
    patch = finalize(state)
    assert patch["messages"][0].content == "FULL CANONICAL REPORT"


def test_research_turn_cannot_pollute_personal_facts(monkeypatch) -> None:
    """Topic facts returned by synthesis never enter user-profile memory."""
    class Structured:
        def invoke(self, messages):
            return {"parsed": finalizer.FinalResponse(
                answer="Research answer", facts=["LangGraph uses checkpoints"]), "raw": None}

    class FakeModel:
        def with_structured_output(self, schema, include_raw=True):
            return Structured()

    monkeypatch.setattr(finalizer, "model", lambda: FakeModel())
    state = {"messages": [HumanMessage("Research LangGraph memory approaches")],
             "active_agent": "research_task", "turn_id": "t1",
             "agent_outputs": [], "monitor_log": []}
    patch = finalize(state)
    assert patch["new_memories"] == []


def test_numeric_note_reference_uses_checkpointed_list(monkeypatch) -> None:
    """A numbered follow-up resolves canonically without history or an LLM."""
    monkeypatch.setattr(finalizer, "model", lambda: (_ for _ in ()).throw(
        AssertionError("LLM must not run")))
    notes = [{"id": f"n{i}", "title": f"Note {i}", "content": f"Content {i}"}
             for i in range(1, 6)]
    state = {"messages": [HumanMessage("4")], "active_agent": "chat", "turn_id": "t2",
             "last_note_list": notes}
    patch = finalize(state)
    assert patch["messages"][0].content == "Note 4\n\nContent 4"


def test_parallel_join_allows_one_approved_branch() -> None:
    """One successful parallel branch clears sibling failure and continues."""
    state = {"turn_id": "t1", "active_agent": "research_task", "error": "research failed",
             "monitor_log": [
                 {"turn_id": "t1", "agent": "research", "approved": False},
                 {"turn_id": "t1", "agent": "task", "approved": True},
             ]}
    assert join(state)["error"] is None
    assert after_join(state, "research_task_writer") == "final"


def test_parallel_join_fails_when_every_branch_fails() -> None:
    """Fan-in remains a hard failure when no specialist result was approved."""
    state = {"turn_id": "t1", "active_agent": "research_task", "error": "both failed",
             "monitor_log": [
                 {"turn_id": "t1", "agent": "research", "approved": False},
                 {"turn_id": "t1", "agent": "task", "approved": False},
             ]}
    assert join(state) == {}
    assert after_join(state, "research_task_writer") == "error"


def test_sqlite_threads_are_isolated(tmp_path) -> None:
    """The SQLite checkpointer never mixes messages between thread IDs."""
    from graph import build_graph

    checkpoint = str(tmp_path / "checkpoints.sqlite")
    memories = str(tmp_path / "memories.sqlite")
    with runtime(checkpoint, memories) as (store, saver):
        app = build_graph(store, saver)
        app.invoke({"messages": [HumanMessage("hello")], "user_id": "u1"},
                   thread_config("u1", "one"))
        app.invoke({"messages": [HumanMessage("hey")], "user_id": "u2"},
                   thread_config("u2", "two"))
        one = app.get_state(thread_config("u1", "one")).values["messages"]
        two = app.get_state(thread_config("u2", "two")).values["messages"]
    assert any(message.content == "hello" for message in one)
    assert all(message.content != "hey" for message in one)
    assert any(message.content == "hey" for message in two)
