import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import build_smoke_graph  


def make_initial_state(text: str) -> dict:
    return {
        "messages": [{"role": "user", "content": text}],
        "user_id": "test-user-1",
        "active_agent": None,
        "task_queue": [],
        "completed_tasks": [],
        "memories": [],
        "summary": "",
        "error": None,
    }


def test_messages_reducer_appends_not_overwrites():
    """add_messages reducer must append new messages, not replace history."""
    graph = build_smoke_graph()
    result = graph.invoke(make_initial_state("hello"), config={"recursion_limit": 20})

    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "hello"
    assert "passthrough received" in result["messages"][1].content


def test_error_field_is_cleared_by_successful_node():
    """The error field must be reset to None after a node completes successfully."""
    state = make_initial_state("test")
    state["error"] = "previous failure"

    graph = build_smoke_graph()
    result = graph.invoke(state, config={"recursion_limit": 20})

    assert result["error"] is None


def test_completed_tasks_reducer_is_additive():
    """completed_tasks uses operator.add, so writes must accumulate, not overwrite."""
    from state import GlobalState
    import operator

    reducer = GlobalState.__annotations__["completed_tasks"].__metadata__[0]
    assert reducer is operator.add
    assert reducer(["task-1"], ["task-2"]) == ["task-1", "task-2"]