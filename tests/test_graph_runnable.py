"""Test: graph must be runnable via both invoke() and stream() (spec constraint)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import build_smoke_graph  # noqa: E402

STATE = {
    "messages": [{"role": "user", "content": "hello smartdesk"}],
    "user_id": "test-user-1",
    "active_agent": None,
    "task_queue": [],
    "completed_tasks": [],
    "memories": [],
    "summary": "",
    "error": None,
}


def test_graph_runs_via_invoke():
    result = build_smoke_graph().invoke(STATE, config={"recursion_limit": 20})
    assert "passthrough received" in result["messages"][-1].content


def test_graph_runs_via_stream():
    events = list(build_smoke_graph().stream(STATE, config={"recursion_limit": 20}))
    assert len(events) == 1
    assert "passthrough" in events[0]