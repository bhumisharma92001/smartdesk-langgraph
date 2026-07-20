"""Pytest suite for memory/checkpointer.py's SQLite thread persistence.

Run with: pytest tests/test_checkpointer.py -v
No API keys required -- this exercises pure LangGraph checkpointing
mechanics with a minimal non-LLM node, not any real agent. Uses
pytest's tmp_path fixture so no database file is left in the repo.
"""
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from memory.checkpointer import build_sqlite_checkpointer, make_thread_config
from state import GlobalState


def _ack_node(state: GlobalState) -> dict:
    """Minimal node: acknowledges the last message, giving checkpointing
    something real to persist without needing any LLM call."""
    last = state["messages"][-1].content if state["messages"] else ""
    return {"messages": [{"role": "assistant", "content": f"ack:{last}"}], "error": None}


def _build_test_graph(checkpointer):
    builder = StateGraph(GlobalState)
    builder.add_node("ack", _ack_node)
    builder.add_edge(START, "ack")
    builder.add_edge("ack", END)
    return builder.compile(checkpointer=checkpointer)


def _fresh_state(text: str) -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "user_id": "test-user-1",
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


def test_same_thread_resumes(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite"
    with build_sqlite_checkpointer(str(db_path)) as checkpointer:
        graph = _build_test_graph(checkpointer)
        config = make_thread_config("thread-resume-test")

        graph.invoke(_fresh_state("first"), config=config)
        result = graph.invoke({"messages": [HumanMessage(content="second")]}, config=config)

        thread_one_messages = result["messages"]
        assert len(thread_one_messages) == 4
        assert [m.content for m in thread_one_messages] == [
            "first", "ack:first", "second", "ack:second",
        ]


def test_different_threads_are_isolated(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite"
    with build_sqlite_checkpointer(str(db_path)) as checkpointer:
        graph = _build_test_graph(checkpointer)

        config_one = make_thread_config("thread-one")
        config_two = make_thread_config("thread-two")

        graph.invoke(_fresh_state("first"), config=config_one)
        graph.invoke({"messages": [HumanMessage(content="second")]}, config=config_one)
        graph.invoke(_fresh_state("isolated"), config=config_two)

        thread_one_messages = graph.get_state(config_one).values["messages"]
        thread_two_messages = graph.get_state(config_two).values["messages"]

        assert len(thread_one_messages) == 4
        assert len(thread_two_messages) == 2

        thread_one_contents = [m.content for m in thread_one_messages]
        thread_two_contents = [m.content for m in thread_two_messages]

        assert "isolated" not in thread_one_contents
        assert "first" not in thread_two_contents
        assert thread_two_contents == ["isolated", "ack:isolated"]


def test_state_survives_database_reopen(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite"
    config = make_thread_config("thread-persist-test")

    # First "process": write state, then close the connection entirely.
    with build_sqlite_checkpointer(str(db_path)) as checkpointer:
        graph = _build_test_graph(checkpointer)
        graph.invoke(_fresh_state("first"), config=config)
        graph.invoke({"messages": [HumanMessage(content="second")]}, config=config)

    # Simulate a fresh process: brand new checkpointer, same db file on disk.
    with build_sqlite_checkpointer(str(db_path)) as checkpointer:
        graph = _build_test_graph(checkpointer)
        state = graph.get_state(config)

        assert state.values, "Expected state to be non-empty after reopening the database"
        restored_messages = state.values["messages"]
        assert len(restored_messages) == 4
        assert restored_messages[-1].content == "ack:second"


def test_thread_config_sets_recursion_limit():
    config = make_thread_config("thread-x", recursion_limit=20)
    assert config["recursion_limit"] == 20
    assert config["configurable"]["thread_id"] == "thread-x"


def test_thread_config_default_recursion_limit_is_20():
    """recursion_limit defaults to 20 even if the caller doesn't pass it."""
    config = make_thread_config("thread-x")
    assert config["recursion_limit"] == 20
