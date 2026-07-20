"""Manual live check for TaskAgent -- real OpenRouter calls, no mocks.
Run directly: python test_task.py
Requires OPENROUTER_API_KEY set in .env.

Covers the same risk categories WriterAgent's test caught real bugs in:
  1. A plain task creation actually persists via create_task.
  2. Marking a step done: get_task must be called before mark_step_done
     (never guess a step_index), and create_task must NOT be called
     again for a task that already exists.
  3. Completing every step: completed_tasks must show up in the patch,
     via GlobalState's operator.add reducer.
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from agents.task_agent import task_agent_node
from state import GlobalState


def make_state(text: str) -> GlobalState:
    return GlobalState(
        messages=[HumanMessage(content=text)],
        user_id="test-user-1",
        active_agent=None,
        task_queue=[],
        completed_tasks=[],
        summary="",
        research_summary="",
        writer_output="",
        task_output="",
        routing_log=[],
        error=None,
    )


def print_trace(messages) -> None:
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  [TOOL CALL] {tc['name']}({tc['args']})")
        elif isinstance(m, ToolMessage):
            tag = "  <ERROR>" if m.status == "error" else ""
            print(f"  [TOOL RESULT{tag}] {m.name}: {str(m.content)[:200]}")
        elif isinstance(m, AIMessage) and m.content:
            print(f"  [AI FINAL] {str(m.content)[:300]}")


def _successful_tool_turns(messages) -> dict[str, int]:
    """Map each tool name to the AI-turn index of its first SUCCESSFUL
    result, skipping error ToolMessages.

    This matters because mark_step_done can legitimately be rejected on
    a first attempt (see tools/task.py's get_task-must-resolve-first
    guard) and then succeed on a retry. Looking only at first
    occurrence -- success or not -- would misread that correct
    self-correction as a same-turn parallel-call violation.
    """
    turn_index = -1
    first_success_turn: dict[str, int] = {}
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            turn_index += 1
        elif isinstance(m, ToolMessage) and m.status != "error":
            if m.name not in first_success_turn:
                first_success_turn[m.name] = turn_index
    return first_success_turn


def _assert_sequential(messages, first: str, second: str) -> None:
    """Assert `first`'s SUCCESSFUL result occurred in a strictly earlier
    AI turn than `second`'s SUCCESSFUL result -- not just an earlier
    position in the flattened message list, and not the same turn
    (which would mean both were requested in parallel before either
    result was available)."""
    successes = _successful_tool_turns(messages)
    assert first in successes, f"Expected a successful {first} call"
    assert second in successes, f"Expected a successful {second} call"
    assert successes[first] < successes[second], (
        f"{first} succeeded in turn {successes[first]} but {second} "
        f"succeeded in turn {successes[second]} -- expected {first} to "
        f"succeed strictly before {second}, not in the same or a later turn."
    )


def case_plain_create() -> None:
    """No existing task. Expects create_task to be called once and
    task_output to be populated, with no error."""
    print("=== Case 1: plain task creation ===")
    store = InMemoryStore()
    state = make_state(
        "I need to publish a blog post. Break it into steps: write a "
        "draft, get it reviewed, and then publish it. Create this as a task."
    )
    patch = task_agent_node(state, store=store)

    print_trace(patch.get("messages", []))
    print("task_output:", patch.get("task_output"))
    print("error:", patch.get("error"))
    assert patch["error"] is None, f"Expected no error, got: {patch['error']}"
    assert patch.get("task_output"), "Expected task_output to be populated"

    tool_names = [
        m.name for m in patch.get("messages", []) if isinstance(m, ToolMessage)
    ]
    assert tool_names.count("create_task") == 1, (
        f"Expected exactly 1 create_task call, got {tool_names.count('create_task')}"
    )

    saved = list(store.search(("tasks", "test-user-1")))
    assert saved, "Expected create_task to have written a task"
    assert len(saved[0].value["steps"]) == 3, "Expected 3 steps to be persisted"
    print(">>> PASS: plain task creation persisted end-to-end.")


def case_mark_step_uses_get_task_first() -> None:
    """Pre-seeds a task directly in the store, then asks to mark one step
    done. Expects get_task to be called BEFORE mark_step_done (never a
    guessed index), and expects create_task to NOT be called again for
    an already-existing task."""
    print("=== Case 2: mark_step_done calls get_task first, no re-create ===")
    store = InMemoryStore()
    from memory.store import SmartDeskStore

    task_id = SmartDeskStore(store).create_task(
        user_id="test-user-1",
        title="Publish blog post",
        steps=["Write draft", "Get it reviewed", "Publish it"],
    )

    state = make_state(
        f"For task {task_id}, mark the 'Write draft' step as done."
    )
    patch = task_agent_node(state, store=store)

    print_trace(patch.get("messages", []))
    print("task_output:", patch.get("task_output"))
    print("error:", patch.get("error"))
    assert patch["error"] is None, f"Expected no error, got: {patch['error']}"

    tool_names = [
        m.name for m in patch.get("messages", []) if isinstance(m, ToolMessage)
    ]
    assert "get_task" in tool_names, "Expected get_task to be called before marking a step done"
    assert "mark_step_done" in tool_names, "Expected mark_step_done to be called"

    _assert_sequential(patch.get("messages", []), "get_task", "mark_step_done")

    assert "create_task" not in tool_names, (
        "create_task was called for a task that already exists -- this "
        "would create a duplicate, unrelated task instead of updating "
        "the existing one."
    )

    updated = SmartDeskStore(store).get_task("test-user-1", task_id)
    assert updated["steps"][0]["done"] is True, "Expected step 0 to be marked done"
    assert updated["status"] == "in_progress", "Expected task to remain in_progress (2 steps left)"

    all_tasks = list(store.search(("tasks", "test-user-1")))
    assert len(all_tasks) == 1, (
        f"Expected exactly 1 task in the store, found {len(all_tasks)} -- "
        f"the model likely created a duplicate task."
    )
    print(">>> PASS: mark_step_done correctly confirmed the index first, no duplicate task.")


def case_completing_all_steps_populates_completed_tasks() -> None:
    """Pre-seeds a task with all but one step already done, then asks to
    finish the last step. Expects the resulting patch to include
    completed_tasks (via the operator.add reducer)."""
    print("=== Case 3: completing the last step populates completed_tasks ===")
    store = InMemoryStore()
    from memory.store import SmartDeskStore

    smartdesk_store = SmartDeskStore(store)
    task_id = smartdesk_store.create_task(
        user_id="test-user-1",
        title="Publish blog post",
        steps=["Write draft", "Get it reviewed", "Publish it"],
    )
    smartdesk_store.mark_step_done("test-user-1", task_id, 0)
    smartdesk_store.mark_step_done("test-user-1", task_id, 1)

    state = make_state(
        f"For task {task_id}, mark the final 'Publish it' step as done."
    )
    patch = task_agent_node(state, store=store)

    print_trace(patch.get("messages", []))
    print("task_output:", patch.get("task_output"))
    print("completed_tasks in patch:", patch.get("completed_tasks"))
    print("error:", patch.get("error"))
    assert patch["error"] is None, f"Expected no error, got: {patch['error']}"

    assert "completed_tasks" in patch, (
        "Expected completed_tasks to be present in the patch once the "
        "task reaches completed status"
    )
    assert len(patch["completed_tasks"]) == 1, "Expected exactly 1 newly-completed task"
    assert patch["completed_tasks"][0]["status"] == "completed"
    assert patch["completed_tasks"][0]["task_id"] == task_id

    _assert_sequential(patch.get("messages", []), "get_task", "mark_step_done")

    updated = smartdesk_store.get_task("test-user-1", task_id)
    assert updated["status"] == "completed", "Expected task status to be completed in the store"
    print(">>> PASS: completing the last step surfaced completed_tasks correctly.")


def case_nonexistent_task_error_surfaces() -> None:
    """Asks the agent to act on a task_id that doesn't exist. get_task
    will fail with a ToolExecutionError; the model may respond with an
    ordinary apologetic AIMessage instead of a raw tool error, but the
    failure must still surface via patch["error"] -- not be silently
    swallowed just because the run's final message looks like normal
    conversation."""
    print("=== Case 4: nonexistent task_id surfaces through error ===")
    store = InMemoryStore()
    fake_task_id = "00000000-0000-0000-0000-000000000000"
    state = make_state(
        f"For task {fake_task_id}, mark the first step as done."
    )
    patch = task_agent_node(state, store=store)

    print_trace(patch.get("messages", []))
    print("task_output:", patch.get("task_output"))
    print("error:", patch.get("error"))
    assert patch["error"] is not None, (
        "Expected error to be set for a nonexistent task_id, got None -- "
        "the underlying get_task failure was not surfaced."
    )
    print(">>> PASS: nonexistent task_id correctly surfaced through error.")


if __name__ == "__main__":
    import sys

    cases = [
        ("Case 1", case_plain_create),
        ("Case 2", case_mark_step_uses_get_task_first),
        ("Case 3", case_completing_all_steps_populates_completed_tasks),
        ("Case 4", case_nonexistent_task_error_surfaces),
    ]
    results = {}

    for name, fn in cases:
        try:
            fn()
            results[name] = "PASS"
        except AssertionError as exc:
            results[name] = f"FAIL: {exc}"
        except Exception as exc:
            results[name] = f"ERROR: {exc}"
        print()

    print("=== Summary ===")
    for name, status in results.items():
        print(f"{name}: {status}")

    if any(status != "PASS" for status in results.values()):
        sys.exit(1)
