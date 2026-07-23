"""Fast regression tests for SmartDesk's critical local behavior."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import ToolException
from langgraph.store.memory import InMemoryStore
from types import SimpleNamespace

import finalizer
from finalizer import finalize
from graph import failed, fan_out, join
from memory.checkpointer import runtime, thread_config
from memory.context import trim_history
from memory.logic import load_memories
from state import merge_error, merge_tasks, reset_or_add
import supervisor
from supervisor import after_agent, after_join, is_greeting, plan_workflow, supervise
from tools.calculator import calculator
from tools.documents import revise_document
from tools.notes import list_notes, save_note
from tools.tasks import create_task
from tools.schemas import CreateTaskInput, WebSearchInput


def test_llm_planner_handles_keyword_free_task(monkeypatch) -> None:
    """Nuanced intent is handled by the structured LLM planner."""
    class Structured:
        def invoke(self, messages):
            return supervisor.Plan(workflow="task")

    class FakeModel:
        def with_structured_output(self, schema):
            return Structured()

    monkeypatch.setattr(supervisor, "model", lambda: FakeModel())
    assert plan_workflow("I need to get organized before the site goes live—break it down for me") == "task"


def test_llm_planner_failure_routes_to_error(monkeypatch) -> None:
    """A provider failure is explicit instead of masquerading as a chat decision."""
    monkeypatch.setattr(supervisor, "model", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    patch = supervise({"messages": [HumanMessage("create a task checklist")]})
    assert patch["active_agent"] is None
    assert patch["error"] == "Planner failed: RuntimeError"
    assert supervisor.planner_status(patch) == "error"


def test_greeting_skips_planner(monkeypatch) -> None:
    """Pure greetings route to chat without an LLM classification call."""
    monkeypatch.setattr(supervisor, "plan_turn", lambda text, active=False: (_ for _ in ()).throw(
        AssertionError("planner must not run")))
    assert supervise({"messages": [HumanMessage("hello!!!")]})["active_agent"] == "chat"


def test_note_save_is_idempotent_by_normalized_title() -> None:
    """Repeated note calls with cosmetic title changes return one canonical ID."""
    store = InMemoryStore()
    first = save_note.func("LangGraph Memory", "first", ["ai"], "u1", store)
    second = save_note.func(" langgraph-memory! ", "second", ["ai"], "u1", store)
    assert second == first
    assert len(store.search(("notes", "u1"))) == 1


def test_note_can_be_retrieved_by_exact_id() -> None:
    """A note lookup is grounded in the store rather than conversation history."""
    store = InMemoryStore()
    note_id = save_note.func("Pricing", "Canonical content", [], "u1", store)
    notes = list_notes.func(None, note_id, "u1", store)
    assert notes == [{"id": note_id, "title": "Pricing",
                      "content": "Canonical content", "tags": []}]


def test_task_rejects_identifier_absent_from_visible_context() -> None:
    """A task cannot persist an LLM-fabricated UUID in its text."""
    invented = "d8512995-8b33-42e2-9954-2e0770851b7c"
    with pytest.raises(ToolException):
        create_task.func("Review", [f"Read note {invented}"],
                         [HumanMessage("Create a review task")], "u1", InMemoryStore())


def test_supervisor_resets_turn_transients(monkeypatch) -> None:
    """Each new turn receives an ID and clears stale transient fields."""
    monkeypatch.setattr(supervisor, "plan_turn",
                        lambda text, active=False: supervisor.Plan(workflow="chat"))
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


def test_parallel_fan_out_is_a_true_no_op() -> None:
    """Starting parallel branches never runs join/error-clearing logic."""
    assert fan_out({"error": "unchanged"}) == {}


def test_revision_requires_exact_document_id() -> None:
    """Writer tools never guess which persisted document to revise."""
    with pytest.raises(ToolException):
        revise_document.func("missing", "shorten it", "u1", InMemoryStore())


def test_unremovable_messages_are_not_summarized() -> None:
    """Messages without IDs remain in context instead of being duplicated into a summary."""
    state = {"messages": [HumanMessage(str(index)) for index in range(13)]}
    assert trim_history(state) == {}


def test_chat_followup_returns_canonical_document(monkeypatch) -> None:
    """A 'share here' follow-up returns stored content without regenerating it."""
    monkeypatch.setattr(finalizer, "model", lambda: (_ for _ in ()).throw(
        AssertionError("LLM must not run")))
    state = {"messages": [HumanMessage("share here")], "active_agent": "show_document",
             "turn_id": "t2", "current_document": {
                 "doc_id": "d1", "content": "EXACT SAVED REPORT", "turn_id": "t1"}}
    patch = finalize(state)
    assert patch["messages"][0].content == "EXACT SAVED REPORT"


def test_show_document_without_document_never_recalls_chat_context(monkeypatch) -> None:
    """Missing canonical documents produce a grounded response without LLM synthesis."""
    monkeypatch.setattr(finalizer, "model", lambda: (_ for _ in ()).throw(
        AssertionError("LLM must not run")))
    state = {"messages": [HumanMessage("show me note 123")],
             "active_agent": "show_document", "turn_id": "t2"}
    patch = finalize(state)
    assert patch["messages"][0].content == "There is no current document to show."


def test_writer_turn_returns_persisted_document_not_preview(monkeypatch) -> None:
    """A new draft's canonical artifact overrides the agent's preview message."""
    monkeypatch.setattr(finalizer, "model", lambda: (_ for _ in ()).throw(
        AssertionError("LLM must not run")))
    state = {"messages": [HumanMessage("write a report")], "active_agent": "writer",
             "turn_id": "t1", "current_document": {
                 "doc_id": "d1", "content": "FULL CANONICAL REPORT", "turn_id": "t1"},
             "agent_outputs": [{"turn_id": "t1", "agent": "writer",
                                 "status": "success",
                                 "content": "Would you like me to share it?"}]}
    patch = finalize(state)
    assert patch["messages"][0].content == "FULL CANONICAL REPORT"


def test_research_turn_cannot_pollute_personal_facts(monkeypatch) -> None:
    """Topic facts returned by synthesis never enter user-profile memory."""
    class FakeModel:
        def invoke(self, messages):
            return AIMessage("Research answer")

    monkeypatch.setattr(finalizer, "model", lambda: FakeModel())
    state = {"messages": [HumanMessage("Research LangGraph memory approaches")],
             "active_agent": "research_task", "turn_id": "t1",
             "agent_outputs": []}
    patch = finalize(state)
    assert "new_memories" not in patch


def test_supervisor_extracts_natural_durable_memory_in_same_call(monkeypatch) -> None:
    """Routing and non-keyword user memory come from one structured model call."""
    class Structured:
        def invoke(self, messages):
            return supervisor.Plan(
                workflow="chat", memories=[supervisor.MemoryFact(
                    type="preference", key="report style", value="concise")])

    class FakeModel:
        def with_structured_output(self, schema):
            return Structured()

    monkeypatch.setattr(supervisor, "model", lambda: FakeModel())
    patch = supervise({"messages": [HumanMessage(
        "When you prepare reports for me, keep them concise.")]})
    assert patch["new_memories"] == ["The user's report style is concise."]


def test_structured_identity_memory_matches_provider_shape() -> None:
    """Provider-style type/key/value memory objects validate and normalize."""
    decision = supervisor.Plan.model_validate({
        "workflow": "chat",
        "memories": [{"type": "identity", "key": "name", "value": "gyi"}],
    })
    assert decision.memories[0].text() == "The user's name is gyi."


def test_chat_memory_retrieves_top_three() -> None:
    """Chat turns retrieve semantic memory instead of skipping user preferences."""
    class Store:
        def search(self, namespace, **kwargs):
            limit = kwargs.get("limit", 10)
            return [SimpleNamespace(value={"text": f"fact {index}"}) for index in range(limit)]

    patch = load_memories({"messages": [HumanMessage("what is my name?")],
                           "user_id": "u1", "active_agent": "chat"}, Store())  # type: ignore[arg-type]
    assert len(patch["memories"]) == 3


def test_parallel_join_allows_one_approved_branch() -> None:
    """One successful parallel branch clears sibling failure and continues."""
    state = {"turn_id": "t1", "active_agent": "research_task", "error": "research failed",
             "agent_outputs": [
                 {"turn_id": "t1", "agent": "research", "status": "failed", "content": ""},
                 {"turn_id": "t1", "agent": "task", "status": "success", "content": "done"},
             ]}
    assert join(state)["error"] is None
    state["error"] = None
    assert after_join(state) == "final"


def test_parallel_join_fails_when_every_branch_fails() -> None:
    """Fan-in remains a hard failure when no specialist result was approved."""
    state = {"turn_id": "t1", "active_agent": "research_task", "error": "both failed",
             "agent_outputs": [
                 {"turn_id": "t1", "agent": "research", "status": "failed", "content": ""},
                 {"turn_id": "t1", "agent": "task", "status": "failed", "content": ""},
             ]}
    assert join(state) == {}
    assert after_join(state) == "error"


def test_supervisor_routes_directly_after_successful_agent() -> None:
    """A successful specialist goes directly to final output without a monitoring node."""
    state = {"turn_id": "t1", "active_agent": "research", "error": None,
             "agent_outputs": [{"turn_id": "t1", "agent": "research",
                                "status": "success", "content": "answer"}]}
    assert after_agent(state, "research_writer") == "final"


def test_sqlite_threads_are_isolated(tmp_path, monkeypatch) -> None:
    """The SQLite checkpointer never mixes messages between thread IDs."""
    from graph import build_graph
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(supervisor, "plan_turn",
                        lambda text, active=False: supervisor.Plan(workflow="chat"))

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
